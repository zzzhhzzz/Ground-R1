
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
import json
import re

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
    parser.add_argument("--prompt_setting", type=str, required=True)

    return parser.parse_args()

args = parse_args()

if args.prompt_setting == "default":
    STAGE_ONE_TEMPLATE = (
        "Question: {Question}\n"
        "Please think about this question as if you were a human pondering deeply. "
        "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions "
        "It's encouraged to include self-reflection or verification in the reasoning process."
        "Give one bounding box coordinate of the region that can help you answer the question better. Following [x1,y1,x2,y2] format."
        "The size of the image: Width:{input_width}, Height:{input_height}. The bounding box you provided should not exceed the image width and height."
        "Provide detailed reasoning between the <think> </think> tags first, then give the bounding box between the <box> </box> tags, finally give your final answer between the <answer> </answer> tags."
        "Format Example: <think> Reasoning process </think><box>[x1,y1,x2,y2]</box><answer> Final answer </answer>"
    )

    STAGE_TWO_TEMPLATE = (
        "Question: {Question}\n"
        "You can see the original image and the cropped image based on the bounding box you provided earlier."
        "You might find the reasoning process, bounding box, and answer you previously provided are not entirely correct or complete."
        "By referring to the original image and the image cropped from your previous bounding box, please try to provide a more accurate bounding box applied to the original image that can help you answer the question better. Following [x1,y1,x2,y2] format. Also, provide your updated reasoning and answer."
        "The size of the original image: Width:{input_width}, Height:{input_height}. The bounding box you provided should not exceed the image width and height."
        "Provide detailed reasoning between the <think> </think> tags first, then give the bounding box between the <box> </box> tags, finally give your final answer between the <answer> </answer> tags."
        "Format Example: <think> Reasoning process </think><box>[x1,y1,x2,y2]</box><answer> Final answer </answer>"
    )
elif args.prompt_setting == "1stage_nothink":
    STAGE_ONE_TEMPLATE = (
    "Question: {Question}\n"
    "Give one bounding box coordinate of the region that can help you answer the question better. Following [x1,y1,x2,y2] format."
    "The size of the image: Width:{input_width}, Height:{input_height}. The bounding box you provided should not exceed the image width and height."
    "Please directly give the bounding box between the <box> </box> tags, and then give your final answer of the question between the <answer> </answer> tags."
    "Format Example: <box>[x1,y1,x2,y2]</box><answer> Final answer </answer>"
    )

    STAGE_TWO_TEMPLATE = (
        "Question: {Question}\n"
        "You can see the original image and the cropped image based on the bounding box you provided earlier."
        "You might find the reasoning process, bounding box, and answer you previously provided are not entirely correct or complete."
        "Please think about these as if you were a human pondering deeply. "
        "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions "
        "It's encouraged to include self-reflection or verification in the reasoning process."
        "By referring to the original image and the image cropped from your previous bounding box, please try to provide a more accurate bounding box applied to the original image that can help you answer the question better. Following [x1,y1,x2,y2] format. Also, provide your updated reasoning and answer."
        "The size of the original image: Width:{input_width}, Height:{input_height}. The bounding box you provided should not exceed the image width and height."
        "Provide detailed reasoning between the <think> </think> tags first, then give the bounding box between the <box> </box> tags, finally give your final answer between the <answer> </answer> tags."
        "Format Example: <think> Reasoning process </think><box>[x1,y1,x2,y2]</box><answer> Final answer </answer>"
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

def _generate_for_stage2(processor, problem_stage2, images, image_stage2, bbox, input_width, input_height):
    stage_two_prompt = {
        "prompt": [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": STAGE_ONE_TEMPLATE.format(Question=problem_stage2, input_width=input_width, input_height=input_height)},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": f"{bbox}"}]},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": STAGE_TWO_TEMPLATE.format(Question=problem_stage2, input_width=input_width, input_height=input_height)},
            ],
        },
        ]
    }
    _prompts = stage_two_prompt.get('prompt')
    prompts_text = [processor.apply_chat_template(_prompts,tokenize=False, add_generation_prompt=True, add_vision_id=True)]
    combined_images = [[images[0], image_stage2]]

    prompt_stage2_inputs = processor(
        text=prompts_text,
        images=combined_images,
        return_tensors="pt",
        padding=True,
        padding_side="left",
        add_special_tokens=False,
    )
    return prompt_stage2_inputs

def run(model_path, rank, args, world_size):
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

        if item["dataset"] == "CLEVR":
            continue

        question = item["problem"]
        image_path = item["image"]
        id = item["problem_id"]
        gt_bbox = item["bboxs"]
        height = item["height"]
        width = item["width"]

        min_pixels = args.min_pixels
        max_pixels = args.max_pixels
        input_height,input_width = smart_resize(height, width, min_pixels=min_pixels, max_pixels=max_pixels)

        messages = [

            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": STAGE_ONE_TEMPLATE.format(Question=question, input_width=input_width, input_height=input_height)},
                ],
            }
        ]

        # prompts_text_failed = processor.apply_chat_template(messages)     ###### 一定要加上add_generation_prompt=True 和训练时对应
        prompts_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, add_vision_id=True)
        
        images = []
        image_inputs, _ = process_vision_info(messages)
        images.append(image_inputs)
        images = images[0]

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

        item["model_answer_stage1"] = output_text[0]
        pattern = r"\[\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\]"
        try:
            bbox = [list(map(float, match)) for match in re.findall(pattern, output_text[0])][0]
        except Exception as e:
            bbox = []

        bbox_to_cal_iou_score = cal_bbox_for_iou(bbox, input_width, input_height, width, height)
        item["bbox_to_cal_iou_score_stage1"] = bbox_to_cal_iou_score

        img_with_bboxes = images[0].copy()
        img_debug_draw = images[0].copy()
        if not bbox:
            pass
        else:
            try:
                bbox = bbox_adjust(bbox, input_width, input_height, width, height, min_size=28)
                if bbox:
                    img_with_bboxes = img_with_bboxes.crop(bbox)
            except:
                pass

        prompt_stage2_inputs = _generate_for_stage2(processor, question, images, img_with_bboxes, output_text[0], input_width, input_height).to(device)
        completion_stage2 = model.generate(**prompt_stage2_inputs, max_new_tokens=512, use_cache=True)

        generated_ids_trimmed_stage2 = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(prompt_stage2_inputs.input_ids, completion_stage2)
        ]

        output_text_stage2 = processor.batch_decode(
            generated_ids_trimmed_stage2, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
    
        pattern = r"\[\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\]"
        try:
            bbox_stage2 = [list(map(float, match)) for match in re.findall(pattern, output_text_stage2[0])][0]
        except Exception as e:
            bbox_stage2 = []

        bbox_to_cal_iou_score_stage2 = cal_bbox_for_iou(bbox_stage2, input_width, input_height, width, height)
        item["bbox_to_cal_iou_score_stage2"] = bbox_to_cal_iou_score_stage2

        item["model_answer_stage2"] = output_text_stage2[0]
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
        output_file = output_file
        with open(output_file, "w", encoding="utf-8") as f:
            for res in gather_list:
                for item in res:  
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

        print(f"{output_file}")
    
if __name__ == "__main__":
    main()