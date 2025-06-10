
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
import re
from collections import defaultdict
import os

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

def compute_iou(gt_bbox, student_bbox):

    x1_gt, y1_gt, x2_gt, y2_gt = gt_bbox
    x1_st, y1_st, x2_st, y2_st = student_bbox

    x1_inter = max(x1_gt, x1_st)
    y1_inter = max(y1_gt, y1_st)
    x2_inter = min(x2_gt, x2_st)
    y2_inter = min(y2_gt, y2_st)

    inter_width = max(0, x2_inter - x1_inter)
    inter_height = max(0, y2_inter - y1_inter)
    inter_area = inter_width * inter_height

    gt_area = (x2_gt - x1_gt) * (y2_gt - y1_gt)
    student_area = (x2_st - x1_st) * (y2_st - y1_st)

    union_area = gt_area + student_area - inter_area

    iou = inter_area / union_area if union_area > 0 else 0

    iou = iou

    return iou


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

        item_to_box = item["solution"]
        image_path = os.path.join(dataset_path, item["image"])
        gt_bbox_xywh = item["bboxs"][0]
        gt_bbox = [gt_bbox_xywh[0], gt_bbox_xywh[1], gt_bbox_xywh[0] + gt_bbox_xywh[2], gt_bbox_xywh[1] + gt_bbox_xywh[3]]
        height = item["height"]
        width = item["width"]

        answers = []
        bbox_to_cal_list = []
        iou_list = []

        final_accuracy = 0

        for desc in item["solution"]:
            messages = [

                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        # {"type": "text", "text": f"Please provide the bounding box coordinate of the region this sentence describes: {item_to_box}. Following [x1,y1,x2,y2] format."},
                        {"type": "text", "text": f"Locate {desc} in this image and output the bbox coordinates in JSON format."},
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
            )
            inputs = inputs.to("cuda")

            input_height = inputs['image_grid_thw'][0][1].item() * 14
            input_width = inputs['image_grid_thw'][0][2].item() * 14

            generated_ids = model.generate(**inputs, max_new_tokens=512, use_cache=True)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            model_answer = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            pattern = r"\[\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\]"
            try:
                bbox_student = [list(map(float, match)) for match in re.findall(pattern, model_answer[0])][0]
            except Exception as e:
                bbox_student = []
            
            if bbox_student:
                bbox_to_cal_iou = cal_bbox_for_iou(bbox_student, input_width, input_height, width, height)
                iou = compute_iou(gt_bbox, bbox_to_cal_iou)
            else:    
                bbox_to_cal_iou = []
                iou = 0

            # print(f"{model_answer[0]}---{gt_bbox}---{iou}")
            answers.append(model_answer[0])
            bbox_to_cal_list.append(bbox_to_cal_iou)
            iou_list.append(iou)
            
            if iou >= 0.5:
                final_accuracy = 1
                break

        item["baseline_answer"] = answers
        item["bbox_to_cal_iou_score_stage2"] = bbox_to_cal_list
        item["iou"] = iou_list
        item["accuracy"] = final_accuracy

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
            dataset = item.get("dataset", "unknown")
            acc = item.get("accuracy", 0)
            dataset_stats[dataset]["correct"] += acc
            dataset_stats[dataset]["total"] += 1

        print("\n=== Dataset Accuracy Summary ===")
        for name, stats in dataset_stats.items():
            avg_acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            print(f"{name}: Accuracy = {avg_acc:.4f}")

        print(f"{output_file}")
    
if __name__ == "__main__":
    main()