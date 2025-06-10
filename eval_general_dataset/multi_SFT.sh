# Generate Answer
export PYTHONPATH="./:$PYTHONPATH"

DATASET_NAME="MMMU_val"

python -m torch.distributed.run \
    --nproc_per_node=4 --master_port=33099 eval_general_dataset/scripts/multi_MME.py \
    --model_path  SFT \
    --eval_dataset eval_general_dataset/dataset/${DATASET_NAME}.jsonl \
    --output_file eval_general_dataset/output/${DATASET_NAME}/SFT.jsonl \
    --min_pixels 3136 \
    --max_pixels 401408 \
    --dataset_path /path/to/LVLM_benchmarks/${DATASET_NAME}/ \
    --dataset_name ${DATASET_NAME} \

## To get MME benchmark score needs another step:
## results_file should be same as the output_file above
# python eval_general_dataset/scripts/cal_MME.py --results_file eval_general_dataset/output/MME/SFT.jsonl