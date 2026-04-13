#!/bin/bash
# Mac 快速训练脚本 - MPS 加速
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=========================================="
echo "  Tarot Card Detection - Mac 快速训练"
echo "=========================================="
echo ""

# 1. 删除旧数据集（标注有bug，必须重建）
if [ -d "$SCRIPT_DIR/scripts/dataset" ]; then
    echo "[1/4] 删除旧数据集..."
    rm -rf "$SCRIPT_DIR/scripts/dataset"
fi

# 2. 生成数据集（减少数量加快速度）
echo "[2/4] 生成合成数据集 (augment=30)..."
cd "$SCRIPT_DIR"
python3 scripts/prepare_data.py --augment 30

# 3. 训练
echo "[3/4] 开始训练..."
python3 scripts/train.py \
    --epochs 50 \
    --imgsz 320 \
    --batch 8 \
    --patience 15 \
    --device mps

# 4. 完成
echo "[4/4] 训练完成！"
echo ""
echo "模型位置: $SCRIPT_DIR/runs/train/weights/best.pt"
echo "ONNX 位置: $SCRIPT_DIR/runs/train/weights/best.onnx"
