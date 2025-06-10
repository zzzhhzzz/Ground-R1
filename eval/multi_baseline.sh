## Generate Answer
export PYTHONPATH="./:$PYTHONPATH"
python -m torch.distributed.run \
    --nproc_per_node=4 --master_port=33099 eval/eval_scripts/multi_baseline.py \
    --model_path  Qwen/Qwen2.5-VL-7B-Instruct \
    --eval_dataset eval/eval_dataset/visualCoT_eval_subset150_choice.jsonl \
    --output_file eval/eval_output/baseline_visualCoT_eval_subset150_choice.jsonl \
    --min_pixels 3136 \
    --max_pixels 401408 \
    --prompt_class only_question

# change prompt_class to simply_answer for more accuracy manual score


### input_file should be same as the output_file above
### Manual Score
python eval/eval_scripts/calculation_baseline.py \
    --input_file eval/eval_output/baseline_visualCoT_eval_subset150_choice.jsonl \

### GPT Score
python eval/eval_scripts/gpt_eval_multi.py \
    --api_key  \
    --api_base  \
    --gpt_model gpt-4o-mini-0718 \
    --input_file eval/eval_output/baseline_visualCoT_eval_subset150_choice.jsonl \
    --output_file eval/eval_output/GPT_Score/baseline_visualCoT_eval_subset150_choice.txt \
    --eval_class baseline


