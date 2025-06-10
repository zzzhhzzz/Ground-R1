export PYTHONPATH="./:$PYTHONPATH"
python -m torch.distributed.run \
    --nproc_per_node=4 --master_port=33099 eval/eval_scripts/multi_0526_new_version.py \
    --model_path  Ground-R1 \
    --eval_dataset eval/eval_dataset/visualCoT_eval_subset150_choice.jsonl \
    --output_file eval/eval_output/GRPO_visualCoT_eval_subset150_choice.jsonl \
    --min_pixels 3136 \
    --max_pixels 401408 \

### input_file should be same as the output_file above
### Manual Score
python eval/eval_scripts/calculation_baseline.py \
    --input_file eval/eval_output/GRPO_visualCoT_eval_subset150_choice.jsonl \

### GPT Score
python eval/eval_scripts/gpt_eval_multi.py \
    --api_key  \
    --api_base  \
    --gpt_model gpt-4o-mini-0718 \
    --input_file eval/eval_output/GRPO_visualCoT_eval_subset150_choice.jsonl \
    --output_file eval/eval_output/GPT_Score/GRPO_visualCoT_eval_subset150_choice.txt \
    --eval_class 0526


