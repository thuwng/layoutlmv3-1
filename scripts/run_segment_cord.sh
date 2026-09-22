#!/bin/bash

set -e

cd /home/s24gbn1/Documents/phg/unilm/layoutlmv3

export PYTHONPATH="/home/s24gbn1/Documents/phg/unilm/layoutlmv3:$PYTHONPATH"
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT="CORD-HIPOS-Experiment"

SEEDS=(42 123 1993)

for SEED in "${SEEDS[@]}"
do

    OUT="./logs/cord-column-bdr-large-v13-seed${SEED}"

    echo ""
    echo "============================================================"
    echo "CORD SEGMENT column-v12- SEED = ${SEED}"
    echo "============================================================"

    # Nếu seed đã có kết quả thì bỏ qua
    if [ -f "$OUT/eval_results.json" ]; then
        echo "[SKIP] Seed ${SEED} đã có kết quả."
        continue
    fi

    rm -rf "$OUT"

    # Giữ nguyên torch.distributed.launch như bản gốc để đảm bảo công bằng 100%
    python examples/run_funsd_cord.py \
    --dataset_name cord \
    --do_train --do_eval \
    --do_predict \
    --use_segment_head \
    --model_name_or_path models/layoutlmv3-large \
    --output_dir "$OUT" \
    --segment_level_layout 1 --visual_embed 1 --input_size 224 \
    --max_steps 1000 --save_steps 1000 --evaluation_strategy steps --eval_steps 100 \
    --learning_rate 5e-5 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 32 \
    --dataloader_num_workers 8 \
    --report_to none \
    --overwrite_output_dir \
    --overwrite_cache \
    --seed "$SEED" \
    --use_hierarchical_position_encoding \
    --max_line_position 80 \
    --max_block_position 15 \
    --use_column_encoding True \
    --max_column_position 8 \
    --use_intra_line_boundary True \
    --lambda_bound_init 0.1
    
done

echo ""
echo "============================================================"
echo "CALCULATING SEGMENT CONTEXT 3-SEED MEAN ± STD"
echo "============================================================"

python - <<'PY'
import os
import json
import numpy as np

seeds = [42, 123, 1993]

metrics = [
    "eval_accuracy",
    "eval_f1",
    "eval_precision",
    "eval_recall",
    "eval_loss",
]

results = {m: [] for m in metrics}

for seed in seeds:
    path = "./logs/cord-column-bdr-large-v13-seed{}/eval_results.json".format(seed)

    print("\nSeed {}:".format(seed))

    if not os.path.exists(path):
        print("  [WARNING] Missing:", path)
        continue

    with open(path, "r") as f:
        data = json.load(f)

    for metric in metrics:
        if metric in data:
            value = float(data[metric])
            results[metric].append(value)
            print("  {:18s} = {:.6f}".format(metric, value))

print("\n" + "=" * 70)
print("FINAL HIPOS RESULT: MEAN ± STD")
print("=" * 70)

summary = {}

for metric in metrics:
    values = results[metric]
    if not values:
        print("{:18s}: NO DATA".format(metric))
        continue

    mean = np.mean(values)
    std = np.std(values, ddof=1) if len(values) > 1 else 0.0

    print(
        "{:18s}: {:.4f} ± {:.4f}".format(
            metric,
            mean,
            std
        )
    )

    summary[metric] = {
        "values": values,
        "mean": float(mean),
        "std": float(std),
    }

output_summary_file = "./logs/cord-column-bdr-large-v13-seed{}/cord_hipos-segpos-column-v13_3seed_summary.json".format(seed)
with open(output_summary_file, "w") as f:
    json.dump(summary, f, indent=2)

print("=" * 70)
print("Saved:", output_summary_file)
PY