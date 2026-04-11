#!/usr/bin/env python3
"""
推理测试脚本 — 用训练好的模型在任意图片上测试
用法：
  python predict.py                                 # 用 best.pt 测试 tarot-static 原图
  python predict.py --model runs/train/weights/best.pt
  python predict.py --img /path/to/photo.jpg --conf 0.3
  python predict.py --img /path/to/photo.jpg --conf 0.3 --save
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "tarot-static" / "cards" / "cards"

# 牌号 → 中文名（78 类）
CARD_NAMES = {
    # Major Arcana (T-00 ~ T-21) → class_id 0-21
    0: "愚人", 1: "魔术师", 2: "女祭司", 3: "女皇", 4: "皇帝",
    5: "教皇", 6: "恋人", 7: "战车", 8: "力量", 9: "隐士",
    10: "命运之轮", 11: "正义", 12: "倒吊人", 13: "死神", 14: "节制",
    15: "恶魔", 16: "塔", 17: "星星", 18: "太阳", 19: "审判", 20: "世界",
    21: "（空白/世界背面）",
    # Wands (W-0A=1 ~ W-QU=14) → class_id 22-36
    22: "权杖 Ace", 23: "权杖二", 24: "权杖三", 25: "权杖四", 26: "权杖五",
    27: "权杖六", 28: "权杖七", 29: "权杖八", 30: "权杖九", 31: "权杖十",
    32: "权杖侍从", 33: "权杖骑士", 34: "权杖女皇", 35: "权杖国王",
    # Cups (C-0A=1 ~ C-QU=14) → class_id 37-51
    36: "圣杯 Ace", 37: "圣杯二", 38: "圣杯三", 39: "圣杯四", 40: "圣杯五",
    41: "圣杯六", 42: "圣杯七", 43: "圣杯八", 44: "圣杯九", 45: "圣杯十",
    46: "圣杯侍从", 47: "圣杯骑士", 48: "圣杯女皇", 49: "圣杯国王",
    # Swords (S-0A=1 ~ S-QU=14) → class_id 52-66
    50: "宝剑 Ace", 51: "宝剑二", 52: "宝剑三", 53: "宝剑四", 54: "宝剑五",
    55: "宝剑六", 56: "宝剑七", 57: "宝剑八", 58: "宝剑九", 59: "宝剑十",
    60: "宝剑侍从", 61: "宝剑骑士", 62: "宝剑女皇", 63: "宝剑国王",
    # Pentacles (P-0A=1 ~ P-KI=13, 少 QU) → class_id 67-81?  Wait...
    # P-0A, P-02, P-03, P-04, P-05, P-06, P-07, P-08, P-09, P-10,
    # P-J1, P-J2, P-KI, P-QU = 14 cards
    67: "星币 Ace", 68: "星币二", 69: "星币三", 70: "星币四", 71: "星币五",
    72: "星币六", 73: "星币七", 74: "星币八", 75: "星币九", 76: "星币十",
    77: "星币侍从", 78: "星币骑士", 79: "星币女皇", 80: "星币国王",
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default=None,
                        help="模型路径（默认自动找 best.pt）")
    parser.add_argument("--img",    default=None,
                        help="测试图片路径（默认遍历 tarot-static 原图）")
    parser.add_argument("--conf",   type=float, default=0.5,
                        help="置信度阈值（默认 0.5）")
    parser.add_argument("--save",   action="store_true",
                        help="保存推理结果图")
    args = parser.parse_args()

    # 自动找模型
    if args.model:
        model_path = args.model
    else:
        candidates = [
            PROJECT_ROOT / "runs" / "train" / "weights" / "best.pt",
            PROJECT_ROOT / "runs" / "train" / "weights" / "last.pt",
            PROJECT_ROOT / "models" / "best.onnx",
        ]
        model_path = next((p for p in candidates if p.exists()), None)
        if not model_path:
            print("[ERROR] 未找到训练好的模型，请先运行 python scripts/train.py")
            return

    print(f"[INFO] 加载模型: {model_path}")
    model = YOLO(str(model_path))

    # 确定测试图片
    if args.img:
        imgs = [Path(args.img)]
    else:
        imgs = sorted(SRC_DIR.glob("sm_RWSa-*.webp"))[:10]  # 最多测 10 张

    print(f"[INFO] 测试 {len(imgs)} 张图片，置信度阈值: {args.conf}\n")

    total = 0
    for img_path in imgs:
        results = model.predict(
            source=str(img_path),
            conf=args.conf,
            imgsz=640,
            verbose=False,
            save=args.save,
        )
        result = results[0]
        boxes = result.boxes

        name = img_path.name
        if boxes is None or len(boxes) == 0:
            print(f"[  空] {name}")
            continue

        total += len(boxes)
        for box in boxes:
            cls_id = int(box.cls.item())
            conf   = box.conf.item()
            name_cn = CARD_NAMES.get(cls_id, f"未知({cls_id})")
            print(f"[检出] {name}  →  {name_cn} (置信度: {conf:.2f})")

    print(f"\n总计检出 {total} 个目标")


if __name__ == "__main__":
    main()
