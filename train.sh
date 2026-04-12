#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "  Tarot Card Detector Training"
echo "  项目目录: $SCRIPT_DIR"
echo "=========================================="

# 0. 检查依赖
echo ""
echo "[Step 0] 检查 Python 依赖..."
PYTHON=$(which python3 || which python)
echo "[OK] 使用 Python: $PYTHON"

$PYTHON -m pip show ultralytics &> /dev/null || {
    echo "[INFO] 安装 ultralytics..."
    $PYTHON -m pip install ultralytics pillow numpy --quiet --upgrade
}

# 1. 检查原图
echo ""
echo "[Step 1] 检查塔罗牌原图..."
SRC="$SCRIPT_DIR/tarot-static/cards/cards"
if [ ! -d "$SRC" ]; then
    echo "[ERROR] 原图目录不存在: $SRC"
    exit 1
fi
IMG_COUNT=$(find "$SRC" -name "sm_RWSa-*.webp" | wc -l | tr -d ' ')
echo "[OK] 找到 $IMG_COUNT 张原图"

# 2. 生成数据集
echo ""
echo "[Step 2] 生成合成训练数据集..."
$PYTHON "$SCRIPT_DIR/scripts/prepare_data.py" --augment 30 --val-split 0.15

# 3. 开始训练
echo ""
echo "[Step 3] 开始训练..."
$PYTHON "$SCRIPT_DIR/scripts/train.py" \
    --model "$SCRIPT_DIR/yolo11n.pt" \
    --epochs 200 \
    --imgsz 640 \
    --batch 32 \
    --patience 50

echo ""
echo "=========================================="
echo "  训练完成！"
echo "=========================================="
echo "  最佳模型: $SCRIPT_DIR/runs/train/weights/best.pt"
echo ""
echo "  测试模型: python3 $SCRIPT_DIR/scripts/predict.py"
echo "=========================================="
