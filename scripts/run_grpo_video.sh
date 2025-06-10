
cd r1-v

python train_data/prepare_data.py --prefix path/to/Visual-CoT/cot_image_data/

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DEBUG_MODE="true"
export LOG_PATH="outputs/2stage_rethink/debug_log_7b.txt"
export DATA_SELECT="true"
export DATA_SELECT_BBOX="true" 
export ACCURACY_PATH="outputs/2stage_rethink/accuracy_select.jsonl"
export BBOX_PATH="outputs/2stage_rethink/bbox_select.jsonl"


CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12352" \
    src/open_r1/grpo.py \
    --output_dir "outputs/2stage_rethink" \
    --model_name_or_path "Qwen/Qwen2.5-VL-7B-Instruct" \
    --dataset_name "train_data/train_33k.jsonl" \
    --deepspeed local_scripts/zero3.json \
    --max_prompt_length 8192 \
    --max_completion_length 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-6 \
    --logging_steps 1 \
    --bf16 \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs 1 \
    --run_name stage2_rethink \
    --save_steps 50 \
    --save_total_limit 2 \
    --max_grad_norm 5 \
    --save_only_model false \
    --num_generations_stage1 2 \
    --num_generations 8   # number of outputs G in grpo, reduce it would lead to faster training and smaller memory cost but higher variance  
# --debug underflow_overflow \
# --max_pixels 401408 \
# --num_generations_stage1 default 8