#!/usr/bin/env python3
"""
塔罗牌检测模型训练脚本
用法：
  python train.py                        # 默认参数训练
  python train.py --epochs 300           # 训练 300 轮
  python train.py --resume              # 从上次中断处恢复
  python train.py --device 0            # 指定 GPU
  python train.py --device mps          # Mac M 系列
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Optional

from ultralytics import YOLO

# ──────────────── 路径配置 ────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_YAML    = PROJECT_ROOT / "scripts" / "dataset" / "tarot.yaml"
MODEL_DIR    = PROJECT_ROOT / "models"

def get_device(arg: Optional[str]) -> str:
    """自动检测最佳设备"""
    if arg:
        return arg
    import torch
    if torch.cuda.is_available():
        return "0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def main():
    parser = argparse.ArgumentParser(description="Train Tarot Card Detector")
    parser.add_argument("--model",    default="yolo11n.pt",    help="基础模型（默认 yolo11n.pt）")
    parser.add_argument("--epochs",   type=int,   default=200,  help="训练轮数（默认 200）")
    parser.add_argument("--imgsz",    type=int,   default=640,  help="输入尺寸（默认 640）")
    parser.add_argument("--batch",    type=int,   default=16,   help="批次大小（默认 16）")
    parser.add_argument("--patience", type=int,   default=50,   help="早停耐心值（默认 50）")
    parser.add_argument("--device",   default=None,             help="设备：0 / -1 / cpu / mps（自动检测）")
    parser.add_argument("--resume",   action="store_true",      help="从上次中断处恢复训练")
    parser.add_argument("--resume-from", default=None,         help="从指定 .pt 恢复")
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"[INFO] 使用设备: {device}")

    # 确认数据集存在
    if not DATA_YAML.exists():
        print(f"[ERROR] 数据集配置不存在：{DATA_YAML}")
        print("请先运行：python scripts/prepare_data.py")
        sys.exit(1)

    # 加载模型
    model_path = PROJECT_ROOT / args.model
    if args.resume_from:
        model = YOLO(args.resume_from)
    else:
        model = YOLO(str(model_path))

    print(f"[INFO] 开始训练，基础模型: {args.model}")
    print(f"[INFO] 数据集配置: {DATA_YAML}")
    print(f"[INFO] epochs={args.epochs}, imgsz={args.imgsz}, batch={args.batch}, device={device}")

    results = model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=device,
        project=str(PROJECT_ROOT),
        name="runs/train",
        exist_ok=True,
        pretrained=True,
        optimizer="auto",
        cos_lr=True,
        amp=True,
        plots=True,
        val=True,
        verbose=True,
        # 增强策略（针对小数据集）
        close_mosaic=10,     # 末 10 轮关闭 mosaic
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
        degrees=15.0,
        translate=0.1,
        scale=0.4,
        shear=5.0,
        perspective=0.0002,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
    )

    print("\n[INFO] 训练完成 ✓")
    print(f"[INFO] 最佳模型: {PROJECT_ROOT}/runs/train/weights/best.pt")
    print(f"[INFO] 最后模型: {PROJECT_ROOT}/runs/train/weights/last.pt")

    # 评估
    print("\n[INFO] 评估模型...")
    metrics = model.val(data=str(DATA_YAML))
    print(f"mAP@50:   {metrics.box.map50:.4f}")
    print(f"mAP@50-95: {metrics.box.map:.4f}")

    # 导出 ONNX
    best_pt = PROJECT_ROOT / "runs" / "train" / "weights" / "best.pt"
    if best_pt.exists():
        print("\n[INFO] 导出 ONNX 模型...")
        export_model = YOLO(str(best_pt))
        MODEL_DIR.mkdir(exist_ok=True)
        onnx_path = export_model.export(format="onnx")
        print(f"[INFO] ONNX 已保存: {onnx_path}")
    else:
        print("[WARN] best.pt 不存在，跳过导出")

if __name__ == "__main__":
    main()
