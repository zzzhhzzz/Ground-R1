import json
import argparse

def update_image_paths_inplace(jsonl_path, new_prefix):
    new_prefix = new_prefix.rstrip('/') + '/'

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    updated_lines = []
    for line in lines:
        item = json.loads(line)
        if 'image' in item and item['image'].startswith("path/to/Visual-CoT/cot_image_data/"):
            item['image'] = item['image'].replace("path/to/Visual-CoT/cot_image_data/", new_prefix, 1)
        elif 'image' in item and item['image'].startswith("path/to/Deepeyes_Dataset/"):
            item['image'] = item['image'].replace("path/to/Deepeyes_Dataset/", new_prefix, 1)
        updated_lines.append(json.dumps(item, ensure_ascii=False) + '\n')

    with open(jsonl_path, 'w', encoding='utf-8') as f:
        f.writelines(updated_lines)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update 'image' paths in a hardcoded JSONL file.")
    parser.add_argument('--prefix', type=str, required=True, help='New prefix to replace "path/to/Visual-CoT/cot_image_data/"')

    args = parser.parse_args()
    
    fixed_jsonl_path1 = "train_data/train_33k.jsonl"
    fixed_jsonl_path2 = "train_data/train_deepeyes.jsonl"
    update_image_paths_inplace(fixed_jsonl_path1, args.prefix)
    update_image_paths_inplace(fixed_jsonl_path2, args.prefix)
