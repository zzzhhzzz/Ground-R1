import os
import json
from PIL import Image
import argparse
from tqdm import tqdm

input_jsonl_path = 'r1-v/train_data/train_33k.jsonl'

def adjust_bbox_min_size(bbox, min_size, img_width, img_height):
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

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_width, x2)
    y2 = min(img_height, y2)

    crop_width = x2 - x1
    crop_height = y2 - y1

    if crop_width < min_size:
        extra = min_size - crop_width
        shift_left = min(x1, extra // 2)
        shift_right = min(img_width - x2, extra - shift_left)
        x1 -= shift_left
        x2 += shift_right
        if x2 - x1 < min_size and x1 > 0:
            x1 = max(0, x2 - min_size)

    if crop_height < min_size:
        extra = min_size - crop_height
        shift_up = min(y1, extra // 2)
        shift_down = min(img_height - y2, extra - shift_up)
        y1 -= shift_up
        y2 += shift_down
        if y2 - y1 < min_size and y1 > 0:
            y1 = max(0, y2 - min_size)

    return (x1, y1, x2, y2)


def process_line(line, crop_root):
    try:
        item = json.loads(line)
        image_path = item["image"]
        dataset = item["dataset"]
        bbox = item["bboxs"][0]
        width = item["width"]
        height = item["height"]

        bbox_str_for_name = f"[{int(bbox[0])}, {int(bbox[1])}, {int(bbox[2])}, {int(bbox[3])}]"

        base_name = os.path.basename(image_path)
        crop_dataset_folder = os.path.join(crop_root, dataset)
        os.makedirs(crop_dataset_folder, exist_ok=True)
        crop_name = f"{base_name}###{bbox_str_for_name}.jpg"
        crop_path = os.path.join(crop_dataset_folder, crop_name)

        with Image.open(image_path) as img:
            adjusted_bbox = adjust_bbox_min_size(bbox, 28, width, height)
            cropped = img.crop(adjusted_bbox)
            cropped.save(crop_path)

    except Exception as e:
        print(f"Failed to process line: {e}")
        return None

def main(crop_root):
    os.makedirs(crop_root, exist_ok=True)

    with open(input_jsonl_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)

    with open(input_jsonl_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=total_lines, desc="Processing"):
            process_line(line.strip(), crop_root)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process JSONL with a configurable crop root path.")
    parser.add_argument('--path_to_image_crop', type=str, default='/path/to/Visual-CoT/cot_image_data_crop', help='Root directory for cropped image data.')
    args = parser.parse_args()

    main(args.path_to_image_crop)