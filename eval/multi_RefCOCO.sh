# Generate Answer
export PYTHONPATH="./:$PYTHONPATH"
python -m torch.distributed.run \
    --nproc_per_node=4 --master_port=33099 eval/eval_scripts/multi_RefCOCO.py \
    --model_path  Qwen/Qwen2.5-VL-7B-Instruct \
    --eval_dataset eval/eval_dataset/RefCOCO_eval_all.jsonl \
    --output_file eval/eval_output/RefCOCO/baseline.jsonl \
    --min_pixels 3136 \
    --max_pixels 401408 \
    --dataset_path path/to/RefCOCO_eval/

# /home/meng/models/Qwen2.5-VL-3B-Instruct
