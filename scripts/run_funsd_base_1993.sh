cd /kaggle/working/layoutlmv3-1

export PYTHONPATH="/kaggle/working/layoutlmv3-1:$PYTHONPATH"
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT="FUNSD-Base-Experiment"

PYTHON_CMD="/kaggle/working/miniconda/envs/layoutlmv3/bin/python"
SEED=1993
OUT="./funsd-base-seed${SEED}"

echo "============================================================"
echo "FUNSD BASE - SEED = ${SEED}"
echo "============================================================"

rm -rf "$OUT"

"$PYTHON_CMD" -m torch.distributed.run \
  --nproc_per_node=2 \
  examples/run_funsd_cord.py \
  --dataset_name funsd \
  --do_train \
  --do_eval \
  --do_predict \
  --use_segment_head \
  --model_name_or_path /kaggle/working/layoutlmv3-base-local \
  --output_dir "$OUT" \
  --segment_level_layout 1 \
  --visual_embed 1 \
  --input_size 224 \
  --max_steps 1000 \
  --save_steps 1000 \
  --evaluation_strategy steps \
  --eval_steps 100 \
  --learning_rate 1e-5 \
  --warmup_ratio 0.1 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --dataloader_num_workers 4 \
  --remove_unused_columns False \
  --use_intra_line_boundary True \
  --lambda_bound_init 0.1 \
  --report_to none \
  --run_name "FUNSD-LR-Split-seed${SEED}" \
  --seed "$SEED" \
  --overwrite_output_dir \
  --overwrite_cache \
  --use_hierarchical_position_encoding True \
  --max_line_position 100 \
  --max_block_position 30 \
  --use_column_encoding True \
  --max_column_position 8 \
  --lambda_bound_init 0.01
