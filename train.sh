#!/bin/bash
# ─────────────────────────────────────────────────────────
# Tarot Card Detector — 训练入口脚本
# ─────────────────────────────────────────────────────────
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# PROJECT_ROOT: tarot-train 的上一级目录（即 tarot-deploy）
PROJECT_ROOT="${SCRIPT_DIR%/*}"

echo "=========================================="
echo "  Tarot Card Detector Training"
echo "  项目目录: $PROJECT_ROOT"
echo "=========================================="

# 0. 检查依赖
echo ""
echo "[Step 0] 检查 Python 依赖..."
if ! command -v python3.11 &> /dev/null; then
    echo "[ERROR] 未找到 python3.11，请先安装 Python 3.11"
    exit 1
fi

python3.11 -m pip show ultralytics &> /dev/null || {
    echo "[INFO] 安装 ultralytics..."
    python3.11 -m pip install ultralytics pillow numpy --quiet --upgrade
}

# 1. 检查原图
echo ""
echo "[Step 1] 检查塔罗牌原图..."
SRC="$PROJECT_ROOT/tarot-static/cards"
if [ ! -d "$SRC" ]; then
    echo "[ERROR] 原图目录不存在: $SRC"
    exit 1
fi
IMG_COUNT=$(find "$SRC" -name "sm_RWSa-*.webp" | wc -l | tr -d ' ')
echo "[OK] 找到 $IMG_COUNT 张原图"

# 2. 生成数据集
echo ""
echo "[Step 2] 生成合成训练数据集..."
python3.11 "$SCRIPT_DIR/scripts/prepare_data.py" --augment 100 --val-split 0.15

# 3. 开始训练
echo ""
echo "[Step 3] 开始训练..."
python3.11 "$SCRIPT_DIR/scripts/train.py" \
    --model yolo11n.pt \
    --epochs 200 \
    --imgsz 320 \
    --batch 1 \
    --patience 50

echo ""
echo "=========================================="
echo "  训练完成！"
echo "=========================================="
echo "  最佳模型: $PROJECT_ROOT/runs/train/weights/best.pt"
echo "  ONNX 模型: $PROJECT_ROOT/models/best.onnx"
echo ""
echo "  测试模型: python3 $SCRIPT_DIR/scripts/predict.py"
echo "=========================================="
