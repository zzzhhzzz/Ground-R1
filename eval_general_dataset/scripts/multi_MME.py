
import logging
from tqdm import tqdm
from dist_utils import get_rank, get_world_size, init_distributed_mode, CustomDataset, CustomDataset_JSONL
import argparse
import numpy as np
from transformers import Qwen2VLForConditionalGeneration
import torch
import torch.distributed as dist
import transformers
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
import json
import re
from collections import defaultdict
import os

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def extract_answer_videor1(text):
    pattern = r'<answer>\s*(.*?)\s*</answer>'
    pattern2 = r'<answer>(.*)'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        match2 = re.search(pattern2, text, re.DOTALL)
        if match2:
            return match2.group(1).strip()
        else:
            # return ""
            return text.strip()

def parse_yesorno_ans(pred_ans):
    # pred_ans = pred_ans.strip().lower()
    pred_ans = extract_answer_videor1(pred_ans).lower()
    if pred_ans in ["yes", "no"]:
        return pred_ans
    prefix_pred_ans = pred_ans[:4]
    if "yes" in prefix_pred_ans:
        return "yes"
    elif "no" in prefix_pred_ans:
        return "no"
    else:
        return "other"
    
def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--eval_dataset", type=str, required=True)
    parser.add_argument("--min_pixels", type=int, default=3136, required=True)
    parser.add_argument("--max_pixels", type=int, default=401408, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--prompt_class", type=str, required=True)

    return parser.parse_args()

args = parse_args()

if args.prompt_class == "MME_RealWorld_Lite":
    STAGE_ONE_TEMPLATE = (
        "{problem} Select the best answer to the above multiple-choice question based on the image. Respond with only the letter (A, B, C, D, or E) of the correct option."
    )
elif args.prompt_class == "MMMU_val" or args.prompt_class == "SEED_Bench":
    STAGE_ONE_TEMPLATE = (
        "{problem} Select the best answer to the above multiple-choice question based on the image. Respond with only the letter (A, B, C or D) of the correct option."
    )
elif args.prompt_class == "MMBench_en_dev" or args.prompt_class == "MMBench_en_test":
    STAGE_ONE_TEMPLATE = (
        "{problem} Select the best answer to the above multiple-choice question based on the image. Respond with only the letter (A, B, C or D) of the correct option."
    )
elif args.prompt_class == "MME" or args.prompt_class == "POPE":
    STAGE_ONE_TEMPLATE = (
        # "{problem} Respond with only the yes or no."
        "{problem}"
    )
elif args.prompt_class == "MMVet" or args.prompt_class == "MMVet2":
    STAGE_ONE_TEMPLATE = (
        "{problem}"
    )
elif args.prompt_class == "RealworldQA":
    STAGE_ONE_TEMPLATE = (
        "{problem}"
    )
elif args.prompt_class == "basic_grpo":
    STAGE_ONE_TEMPLATE = (
        "Question: {problem}\n"
        "Please think about this question as if you were a human pondering deeply. "
        "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions "
        "It's encouraged to include self-reflection or verification in the reasoning process."
        "Provide detailed reasoning between the <think> </think> tags first, then give your final answer between the <answer> </answer> tags."
        "Format Example: <think> Reasoning process </think><answer> Final answer </answer>"
    )
elif args.prompt_class == "onestage_nothink":
    STAGE_ONE_TEMPLATE = (
        "Question: {problem}\n"
        "Give one bounding box coordinate of the region that can help you answer the question better. Following [x1,y1,x2,y2] format."
        "The size of the image: Width:{input_width}, Height:{input_height}. The bounding box you provided should not exceed the image width and height."
        "Please directly give the bounding box between the <box> </box> tags, and then give your final answer of the question between the <answer> </answer> tags."
        "Format Example: <box>[x1,y1,x2,y2]</box><answer> Final answer </answer>"
    )

def load_model_and_dataset(model_path, rank, world_size, args):

    eval_dataset = args.eval_dataset
    min_pixels = args.min_pixels
    max_pixels = args.max_pixels

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=('cuda:{}'.format(rank))
        )

    processor = AutoProcessor.from_pretrained(
        model_path, min_pixels=min_pixels, max_pixels=max_pixels
    )
    dataset = CustomDataset_JSONL(eval_dataset)
    dataset.set_rank_and_world_size(rank, world_size)

    return model, processor, dataset


def run(model_path, dataset_path, rank, args, world_size):
    if rank != 0:
        transformers.utils.logging.set_verbosity_error()
        logger.setLevel(transformers.logging.ERROR)

    logger.info(f'loading model and constructing dataset to gpu {rank}...')
    model, processor, dataset = load_model_and_dataset(model_path,
                                                        rank,
                                                        world_size,
                                                        args)

    output_data = []
    done_count = 0
    if rank == 0:
        tbar = tqdm(total=len(dataset))

    for item in dataset:

        problem = item['problem']
        ground_truth = item["solution"]
        image_path = os.path.join(dataset_path, item["Image"])

        messages = [

            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": STAGE_ONE_TEMPLATE.format(problem=problem)},
                ],
            }
        ]


        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, add_vision_id=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            # padding_side="left",
            # add_special_tokens=False,
        )

        inputs = inputs.to("cuda")

        input_height = inputs['image_grid_thw'][0][1].item() * 14
        input_width = inputs['image_grid_thw'][0][2].item() * 14

        generated_ids = model.generate(**inputs, max_new_tokens=512, use_cache=True, do_sample=False)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        model_answer = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        answer = model_answer[0]
        # print(f"{model_answer[0]}")
        item["stage1_answer"] = answer

        try:
            if args.prompt_class == "POPE":
                if parse_yesorno_ans(extract_answer_videor1(answer).lower()) == ground_truth.lower():
                    score = 1
                else:
                    score = 0
            elif args.prompt_class == "RealworldQA":
                if extract_answer_videor1(answer).lower() == ground_truth.lower():
                    score = 1
                else:
                    score = 0
            else:
                if answer[0].lower() == ground_truth.lower():
                    score = 1
                elif extract_answer_videor1(answer)[0].lower() == ground_truth.lower():
                    score = 1
                else:
                    score = 0
        except Exception as e:
            score = 0
        item["score"] = score

        output_data.append(item)

        if rank == 0:
            tbar.update(len(output_data) - done_count)
            done_count = len(output_data)

    return output_data

def main():

    args = parse_args()
    args.distributed = True
    args.dist_url = "env://"

    init_distributed_mode(args)
    rank, world_size = get_rank(), get_world_size()

    model_path = args.model_path
    output_file = args.output_file
    dataset_path = args.dataset_path
    local_result = run(model_path, dataset_path, rank, args, world_size)

    gather_list = [None] * world_size
    dist.all_gather_object(gather_list, local_result)  

    if rank == 0:
        all_items = []
        with open(output_file, "w", encoding="utf-8") as f:
            for res in gather_list:
                for item in res:  
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    all_items.append(item)

        dataset_stats = defaultdict(lambda: {"correct": 0, "total": 0})

        for item in all_items:
            dataset = item.get("Category", "unknown")
            acc = item.get("score", 0)
            dataset_stats[dataset]["correct"] += acc
            dataset_stats[dataset]["total"] += 1

        print("\n=== Dataset Accuracy Summary ===")
        for name, stats in dataset_stats.items():
            avg_acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            print(f"{name}: Accuracy = {avg_acc:.4f}")

        total_correct = sum(stats["correct"] for stats in dataset_stats.values())
        total_count = sum(stats["total"] for stats in dataset_stats.values())
        overall_acc = total_correct / total_count if total_count > 0 else 0
        print(f"Overall Accuracy = {overall_acc:.4f}")
        
        print(f"{output_file}")
    
if __name__ == "__main__":
    main()