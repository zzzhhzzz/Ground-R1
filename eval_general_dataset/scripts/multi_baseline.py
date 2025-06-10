
import logging
from tqdm import tqdm
from dist_utils import get_rank, get_world_size, init_distributed_mode, CustomDataset, CustomDataset_JSONL
import argparse
import numpy as np

import torch
import torch.distributed as dist
import transformers
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
import json

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

STAGE_ONE_TEMPLATE = (
    "Question: {Question}\n"
    "Please think about this question as if you were a human pondering deeply. "
    "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions "
    "It's encouraged to include self-reflection or verification in the reasoning process."
    "Provide detailed reasoning between the <think> </think> tags first, then give your final answer between the <answer> </answer> tags."
    "Format Example: <think> Reasoning process </think><answer> Final answer </answer>"
)
def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--eval_dataset", type=str, required=True)
    parser.add_argument("--min_pixels", type=int, default=3136, required=True)
    parser.add_argument("--max_pixels", type=int, default=401408, required=True)
    parser.add_argument("--prompt_class", type=str, required=True)

    return parser.parse_args()

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

def run(model_path, rank, args, world_size):
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
    prompt_class = args.prompt_class
    if rank == 0:
        tbar = tqdm(total=len(dataset))

    for item in dataset:

        question = item["problem"]
        image_path = item["image"]

        if prompt_class == "only_question":
            messages = [

                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": f"{question}"},
                    ],
                }
            ]
        elif prompt_class == "simply_answer":
            messages = [

                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": f"Please only give the final answer of the Question:{question}"},
                    ],
                }
            ]
        elif prompt_class == "basic_GRPO":
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": STAGE_ONE_TEMPLATE.format(Question=question)},
                    ],
                },
            ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, add_vision_id=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        generated_ids = model.generate(**inputs, max_new_tokens=1000)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        model_answer = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        item["baseline_answer"] = model_answer[0]
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
    local_result = run(model_path, rank, args, world_size)

    gather_list = [None] * world_size
    dist.all_gather_object(gather_list, local_result)  

    if rank == 0:
        with open(output_file, "w", encoding="utf-8") as f:
            for res in gather_list:
                for item in res:  
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

        print(f"{output_file}")
    
if __name__ == "__main__":
    main()