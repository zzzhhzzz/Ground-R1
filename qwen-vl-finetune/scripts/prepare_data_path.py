import argparse
import json

def update_image_paths_inplace(json_path, new_path1, new_path2):
    old_path1 = "/your/path/to/Visual-CoT/cot_image_data"
    old_path2 = "/your/path/to/Visual-CoT/cot_image_data_crop"

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for item in data:
        if 'image' in item:
            item['image'] = [
                img.replace(old_path1, new_path1).replace(old_path2, new_path2)
                for img in item['image']
            ]

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Updated image paths in: {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replace hardcoded paths in 'image' field of JSONL file.")
    parser.add_argument('--path_to_image', type=str, required=True, help='New path to replace cot_image_data')
    parser.add_argument('--path_to_image_crop', type=str, required=True, help='New path to replace cot_image_data_crop')

    args = parser.parse_args()

    fixed_jsonl1 = "qwen-vl-finetune/qwenvl/data/Vanilla_SFT.json"
    fixed_jsonl2 = "qwen-vl-finetune/qwenvl/data/Ground_SFT.json"
    update_image_paths_inplace(fixed_jsonl1, args.path_to_image, args.path_to_image_crop)
    update_image_paths_inplace(fixed_jsonl2, args.path_to_image, args.path_to_image_crop)
