import json
from image_pro import process_vision_info, smart_resize
############################# 更改max_pixel_size!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

def process_jsonl(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            data = json.loads(line)
            width = data.get("width")
            height = data.get("height")
            if width is not None and height is not None:
                input_width, input_height = smart_resize(width, height)
                data["input_width"] = input_width
                data["input_height"] = input_height
            outfile.write(json.dumps(data, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    input_jsonl = "/home/meng/GRPO/src/data/vcot_data_v3/datav1_ciai_filter6.jsonl"   # 替换为你的输入文件路径
    output_jsonl = "/home/meng/GRPO/src/data/vcot_data_v3/datav1_ciai_filter6_size.jsonl" # 替换为你想要的输出文件路径
    process_jsonl(input_jsonl, output_jsonl)
