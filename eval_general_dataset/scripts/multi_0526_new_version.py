
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
from image_pro import process_vision_info, smart_resize
from PIL import Image, ImageDraw
import os
import json
import re
import copy
from collections import defaultdict

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--eval_dataset", type=str, required=True)
    parser.add_argument("--min_pixels", type=int, default=3136, required=True)
    parser.add_argument("--max_pixels", type=int, default=401408, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)

    return parser.parse_args()

args = parse_args()

STAGE_ONE_TEMPLATE = (
    "Question: {Question}\n"
    "Based on the original image and the question, reason whether there exists a region in the image that could help you answer the question better. If such a region exists, provide one bounding box coordinate in the format [x1,y1,x2,y2] inside the <box> and </box> tags."
    "The size of the image: Width:{input_width}, Height:{input_height}. The bounding box you provided should not exceed the image width and height."
    "Then, you will receive a cropped image based on the bounding box. Use both images to continue reasoning inside a new <think> tag. You may conduct multiple rounds of grounding to refine your region as you want. The bounding box you provide should always be selected based on the original image."
    "If at any point you determine no further visual information is needed, you may directly provide the final answer inside the <answer> and </answer> tags."
    "Format Example: <think> Reasoning </think> <box>[x1,y1,x2,y2]</box> OR <think> Reasoning </think> <answer> final answer </answer>"
)

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

def bbox_adjust(bbox, input_width, input_height, width, height, min_size=28):
    x1, y1, x2, y2 = bbox

    x1 = min(x1, input_width)
    x2 = min(x2, input_width)
    y1 = min(y1, input_height)
    y2 = min(y2, input_height)
    
    y1 = int(y1/input_height * height)
    x1 = int(x1/input_width * width)
    y2 = int(y2/input_height * height)
    x2 = int(x2/input_width * width)

    bbox_width = x2 - x1
    bbox_height = y2 - y1
    
    if bbox_width < min_size:
        s = (min_size - bbox_width) // 2
        x1 -= s
        x2 += s
        if x2 - x1 < min_size:
            x2 += 1
    
    if bbox_height < min_size:
        s = (min_size - bbox_height) // 2
        y1 -= s
        y2 += s
        if y2 - y1 < min_size:
            y2 += 1

    bbox_width = x2 - x1
    bbox_height = y2 - y1
    if max(bbox_width, bbox_height) / min(bbox_width, bbox_height) >= 200:
        return []

    return [x1, y1, x2, y2]

def cal_bbox_for_iou(bbox, input_width, input_height, width, height):
    if bbox:
        x1, y1, x2, y2 = bbox
        if x1 >= x2 or y1 >= y2:
            return []
        y1 = int(y1/input_height * height)
        x1 = int(x1/input_width * width)
        y2 = int(y2/input_height * height)
        x2 = int(x2/input_width * width)
        return [x1, y1, x2, y2]
    else:
        return bbox
    
def gt_bbox_adjust(bbox, min_size=28):
    x1, y1, x2, y2 = bbox

    bbox_width = x2 - x1
    bbox_height = y2 - y1
    
    if bbox_width < min_size:
        s = (min_size - bbox_width) // 2
        x1 -= s
        x2 += s
        if x2 - x1 < min_size:
            x2 += 1
    
    if bbox_height < min_size:
        s = (min_size - bbox_height) // 2
        y1 -= s
        y2 += s
        if y2 - y1 < min_size:
            y2 += 1

    return [x1, y1, x2, y2]


def _crop_image_for_next_stage(output_text, origin_image, input_width, input_height, width, height):

    img_with_bboxes = origin_image.copy()

    pattern = r"\[\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\]"
    try:
        bbox = [list(map(float, match)) for match in re.findall(pattern, output_text[0])][0]
    except Exception as e:
        bbox = []

    if not bbox:
        pass
        # print(f"no bbox:{output_text}")
    else:
        bbox = bbox_adjust(bbox, input_width, input_height, width, height, min_size=28)
        if bbox:
            # draw = ImageDraw.Draw(img_with_bboxes)
            # draw.rectangle(bbox, outline="red", width=2)
            img_with_bboxes = img_with_bboxes.crop(bbox)
        
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    # img_box_path = os.path.join("/map-vepfs/caomeng/code/MoBA/Video-CoT/output_r1/debug_img/", f"imagebox{i}_{timestamp}.png")
    # img_with_bboxes.save(img_box_path, "PNG")
    return img_with_bboxes

def _prepare_for_stage2(origin_prompt, origin_problem, input_width, input_height, combined_images, image_stage2, bbox):

    next_stage_entry = {
        "role": "user",
        "content": [
            {"type": "image"},
            {
                "type": "text",
                "text": STAGE_ONE_TEMPLATE.format(Question=origin_problem, input_width=input_width, input_height=input_height)
            },
        ],
    }

    origin_prompt.extend([
    {
        "role": "assistant",
        "content": [{"type": "text", "text": str(bbox)}]
    },
    copy.deepcopy(next_stage_entry)
    ])

    combined_images.append(image_stage2)
    return origin_prompt, combined_images

def _generate_for_stage2(processor, device, messages_stage2, all_images_stage2):
    
    texts = processor.apply_chat_template(messages_stage2, tokenize=False, add_generation_prompt=True, add_vision_id=True)

    prompt_stage2_inputs = processor(
        text=texts,
        images=all_images_stage2,
        return_tensors="pt",
        padding=True,
        padding_side="left",
        add_special_tokens=False,
    ).to(device)
    return prompt_stage2_inputs   

def run(model_path, dataset_path, rank, args, world_size):
    if rank != 0:
        transformers.utils.logging.set_verbosity_error()
        logger.setLevel(transformers.logging.ERROR)

    logger.info(f'loading model and constructing dataset to gpu {rank}...')
    model, processor, dataset = load_model_and_dataset(model_path, rank, world_size, args)

    output_data = []
    done_count = 0
    device = f"cuda:{rank}"

    if rank == 0:
        tbar = tqdm(total=len(dataset))

    for item in dataset:

        problem = item['problem']
        ground_truth = item["solution"]
        image_path = os.path.join(dataset_path, item["Image"])

        with Image.open(image_path) as img:
            width, height = img.size  # PIL (width, height)

        min_pixels = args.min_pixels
        max_pixels = args.max_pixels
        input_height,input_width = smart_resize(height, width, min_pixels=min_pixels, max_pixels=max_pixels)

        generation_list = []

        messages = [

            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": STAGE_ONE_TEMPLATE.format(Question=problem, input_width=input_width, input_height=input_height)},
                ],
            }
        ]

        # prompts_text_failed = processor.apply_chat_template(messages)     ###### 一定要加上add_generation_prompt=True 和训练时对应
        prompts_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, add_vision_id=True)
        
        images = []
        image_inputs, _ = process_vision_info(messages)
        images.append(image_inputs)
        images = images[0]

        prompt_for_generation = messages
        all_images_final_list = copy.deepcopy(images)
        
        prompt_inputs_state1 = processor(
            text=prompts_text,
            images=images,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        ).to(device)

        input_height = prompt_inputs_state1['image_grid_thw'][0][1].item() * 14
        input_width = prompt_inputs_state1['image_grid_thw'][0][2].item() * 14

        width, height = images[0].size

        completion = model.generate(**prompt_inputs_state1, max_new_tokens=512, use_cache=True)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(prompt_inputs_state1.input_ids, completion)
        ]

        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        ### init_generate_end
        generation_list.append(output_text[0])
        iteration = 0
        continue_generate = True
        while continue_generate:
            if ("<answer>" in output_text[0] or iteration == 4):
                continue_generate = False
            else:
                img_with_bboxes_stage2 = _crop_image_for_next_stage(
                    output_text[0], images[0], input_width, input_height, width, height
                )
                prompt_for_generation, all_images_final_list = _prepare_for_stage2(
                    prompt_for_generation, problem, input_width, input_height,
                    all_images_final_list, img_with_bboxes_stage2, output_text[0]
                )
                prompt_stage2_inputs = _generate_for_stage2(processor, device, prompt_for_generation, all_images_final_list)

                completion_stage_next = model.generate(**prompt_stage2_inputs, max_new_tokens=512, use_cache=True)

                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(prompt_stage2_inputs.input_ids, completion_stage_next)
                ]

                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                generation_list.append(output_text[0])
            iteration += 1
        item['generation_list'] = generation_list
        item['iteration'] = iteration
        answer = generation_list[-1]
        if args.dataset_name == "POPE":
            if parse_yesorno_ans(extract_answer_videor1(answer).lower()) == ground_truth.lower():
                score = 1
            else:
                score = 0
        elif args.dataset_name == "RealworldQA":
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