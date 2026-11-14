#!/usr/bin/env bash
# Đóng gói tối thiểu để dựng dữ liệu SFT trên Colab (không kèm audio / environment / zaic_tools).
# Chạy từ thư mục lifemate/:  bash finetune/make_colab_bundle.sh
# Kết quả: finetune/out/lifemate_bundle.zip  -> upload vào repo HF PRIVATE (notebook sẽ hướng dẫn).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p finetune/out
rm -f finetune/out/lifemate_bundle.zip
zip -q finetune/out/lifemate_bundle.zip \
  main/agent.py main/qwen_local.py main/prompt_rules.txt \
  finetune/build_sft.py \
  train/train.jsonl train/train_tools_catalog.json
ls -lh finetune/out/lifemate_bundle.zip
