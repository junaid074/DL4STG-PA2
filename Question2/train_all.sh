#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTORCH_ENABLE_MPS_FALLBACK=1
mkdir -p logs checkpoints

for model in A B C D; do
  for seed in 0 1; do
    ckpt="checkpoints/model${model}_seed${seed}.pt"
    if [[ -f "$ckpt" ]]; then
      echo "skip $ckpt (exists)"
      continue
    fi
    echo "===== train model $model seed $seed ====="
    python3 -u -m q2.train --model "$model" --seed "$seed" --epochs 30 --root . \
      2>&1 | tee "logs/train_${model}_s${seed}.log"
  done
done
echo ALL_TRAINING_DONE
