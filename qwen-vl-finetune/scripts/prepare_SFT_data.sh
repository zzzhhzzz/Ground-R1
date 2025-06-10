
### --path_to_image_crop for cropped images for Ground-SFT
python qwen-vl-finetune/scripts/prepare_2stage_image.py --path_to_image_crop /your/path/to/Visual-CoT/cot_image_data_crop

### --path_to_image_crop should be same as above
### --path_to_image should be the VisCoT Path
python qwen-vl-finetune/scripts/prepare_data_path.py --path_to_image /your/path/to/Visual-CoT/cot_image_data --path_to_image_crop /your/path/to/Visual-CoT/cot_image_data_crop
