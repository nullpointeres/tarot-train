#!/usr/bin/env python3
"""
生成塔罗牌合成训练数据集（YOLO 格式）
78 张原图 + 合成 → 训练集 / 验证集

用法：
  python prepare_data.py
  python prepare_data.py --augment 200   # 每张原图生成 200 张合成图
  python prepare_data.py --augment 500 --val-split 0.2
"""

import argparse
import os
import random
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance

# ──────────────── 路径配置 ────────────────
SRC_DIR = Path(__file__).parent.parent.parent / "tarot-static" / "cards"
OUT_DIR = Path(__file__).parent / "dataset"
IMG_TRAIN = OUT_DIR / "images" / "train"
IMG_VAL   = OUT_DIR / "images" / "val"
LBL_TRAIN = OUT_DIR / "labels" / "train"
LBL_VAL   = OUT_DIR / "labels" / "val"

# ──────────────── 牌号映射 ────────────────
# sm_RWSa-{SUIT}-{RANK}.webp
# SUIT: T=Major, W=Cups, S=Swords, P=Pentacles, C=Cups (wait...)
# RWSa naming: T=Major, W=Wands, C=Cups, S=Swords, P=Pentacles
RANK_ORDER = ["0A", "02", "03", "04", "05", "06", "07", "08", "09", "10",
              "J1", "J2", "KI", "QU"]
SUIT_MAP = {"T": "major", "W": "wands", "C": "cups", "S": "swords", "P": "pentacles"}

# card_no: 0-21 Major(T-00~T-21), 22-36 Wands, 37-51 Cups, 52-66 Swords, 67-77 Pentacles
MAJOR_NAMES = [
    "The Fool", "The Magician", "The High Priestess", "The Empress",
    "The Emperor", "The Hierophant", "The Lovers", "The Chariot",
    "Strength", "The Hermit", "Wheel of Fortune", "Justice",
    "The Hanged Man", "Death", "Temperance", "The Devil",
    "The Tower", "The Star", "The Sun", "Judgment", "The World",
    # T-00 is The Fool, T-21 is The World, T-22 would be... traditionally blank but we use T-21 as World
    # Actually T-00 through T-21 = 22 cards = 0-21
]

def parse_card_info(fname: str):
    """
    从文件名解析 class_id（与数据库 card_no 0-77 完全对齐）
    class_id 0-21  : Major Arcana  T-00~T-21
    class_id 22-36 : Wands         W-0A(1)~W-QU(14)
    class_id 37-51 : Cups          C-0A(1)~C-QU(14)
    class_id 52-66 : Swords        S-0A(1)~S-QU(14)
    class_id 67-77 : Pentacles     P-0A(1)~P-KI(11)  [少 QU，多出 J1/J2 两种]
    """
    parts = fname.replace(".webp", "").split("-")  # sm_RWSa, T, 00
    suit_code = parts[1]
    rank = parts[2]
    suit_name = SUIT_MAP[suit_code]

    if suit_code == "T":
        # T-00(The Fool)→0, T-01(The Magician)→1, ..., T-21(The World)→21
        class_id = int(rank)
    else:
        # 0A=1, 02=2, ..., 10=10, J1=11, J2=12, KI=13, QU=14
        rank_to_num = {"0A": 1, "02": 2, "03": 3, "04": 4, "05": 5,
                       "06": 6, "07": 7, "08": 8, "09": 9, "10": 10,
                       "J1": 11, "J2": 12, "KI": 13, "QU": 14}
        rank_num = rank_to_num[rank]
        suit_offset = {"W": 0, "C": 1, "S": 2, "P": 3}[suit_code]
        class_id = 22 + suit_offset * 15 + rank_num  # 22/37/52/67

    return class_id, suit_name, rank

def build_class_yaml() -> str:
    """生成 tarot.yaml class_names 部分"""
    lines = []
    lines.append("names:")
    # Major 0-21
    major = [
        "The Fool", "The Magician", "The High Priestess", "The Empress",
        "The Emperor", "The Hierophant", "The Lovers", "The Chariot",
        "Strength", "The Hermit", "Wheel of Fortune", "Justice",
        "The Hanged Man", "Death", "Temperance", "The Devil",
        "The Tower", "The Star", "The Moon", "The Sun", "Judgment", "The World",
        # Note: T-00 = The Fool (id 0), T-21 = The World (id 21)
    ]
    for i, name in enumerate(major):
        lines.append(f"  {i}: {name}")

    # Minor Arcana suits in order: Wands, Cups, Swords, Pentacles
    minor_ranks = ["Ace", "Two", "Three", "Four", "Five", "Six", "Seven",
                   "Eight", "Nine", "Ten", "Page", "Knight", "Queen", "King"]
    minor_suits = [("Wands", "wands"), ("Cups", "cups"), ("Swords", "swords"), ("Pentacles", "pentacles")]
    card_no = 22
    for suit_name, _ in minor_suits:
        for rank_name in minor_ranks:
            lines.append(f"  {card_no}: {suit_name} {rank_name}")
            card_no += 1

    return "\n".join(lines)

# ──────────────── 合成函数 ────────────────

def random_background():
    """生成随机背景图"""
    w, h = 1280, 1280
    # 随机底色：深色系为主（模拟真实场景）
    palette = [
        (20, 15, 35),   # 深紫
        (10, 20, 30),   # 深蓝
        (25, 15, 20),   # 深红
        (15, 25, 20),   # 深绿
        (30, 25, 20),   # 暖棕
        (40, 35, 30),   # 米灰
    ]
    base = random.choice(palette)
    arr = np.random.randint(0, 30, (h, w, 3), dtype=np.uint8)
    arr[:, :] += np.array(base, dtype=np.uint8)
    # 添加噪点
    noise = np.random.randint(-15, 15, (h, w, 3), dtype=np.int16)
    arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")

def paste_card(bg: Image, card_img: Image, target_w: int, target_h: int,
               center: tuple[int, int], angle: float,
               brightness: float, contrast: float) -> list:
    """
    将卡牌图合成到背景上，返回 YOLO bbox [cx, cy, w, h]（归一化）
    """
    # 缩放
    scale = random.uniform(0.18, 0.35)
    new_w = int(card_img.width * scale)
    new_h = int(card_img.height * scale)

    # 旋转
    rotated = card_img.resize((new_w, new_h), Image.LANCZOS)
    if abs(angle) > 0.5:
        rotated = rotated.rotate(angle, expand=True, fillcolor=(255, 255, 255))

    # 亮度和对比度调整
    enhancer_b = ImageEnhance.Brightness(rotated)
    rotated = enhancer_b.enhance(brightness)
    enhancer_c = ImageEnhance.Contrast(rotated)
    rotated = enhancer_c.enhance(contrast)

    # 中心坐标
    cx, cy = center
    paste_x = cx - rotated.width // 2
    paste_y = cy - rotated.height // 2

    bg.paste(rotated, (paste_x, paste_y), rotated if rotated.mode == "RGBA" else None)

    # 计算原始卡区域的 bbox（相对于背景尺寸）
    bbox = [
        center[0] / 1280,
        center[1] / 1280,
        rotated.width / 1280,
        rotated.height / 1280,
    ]
    return bbox

def synth_one(card_path: Path, class_id: int, idx: int, out_img_dir: Path, out_lbl_dir: Path):
    """生成一张合成图"""
    card_img = Image.open(card_path).convert("RGBA")

    bg = random_background()
    bg_w, bg_h = 1280, 1280

    # 随机放置 1-2 张牌（1 张为主，偶尔 2 张增加多样性）
    num_cards = 1 if random.random() < 0.75 else 2
    bboxes = []

    for k in range(num_cards):
        angle = random.uniform(-25, 25) if num_cards > 1 else random.uniform(-12, 12)
        brightness = random.uniform(0.75, 1.25)
        contrast   = random.uniform(0.85, 1.20)

        # 边缘留白
        margin = int(bg_w * 0.15)
        cx = random.randint(margin, bg_w - margin)
        cy = random.randint(margin, bg_h - margin)

        bbox = paste_card(bg, card_img, 0, 0, (cx, cy), angle, brightness, contrast)
        bboxes.append((class_id, bbox))

    # 保存图片（转为 JPEG 节省空间）
    out_name = f"card_{class_id:03d}_{idx:05d}.jpg"
    bg.convert("RGB").save(out_img_dir / out_name, quality=88)

    # 保存标注
    lbl_name = out_name.replace(".jpg", ".txt")
    with open(out_lbl_dir / lbl_name, "w") as f:
        for cid, bb in bboxes:
            cx, cy, w, h = bb
            f.write(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

def copy_originals(out_img_dir: Path, out_lbl_dir: Path, split_name: str):
    """将原图作为无增强的参考图"""
    pass  # 通过合成图已足够，原图通过 augment 参数控制

# ──────────────── 主流程 ────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--augment", type=int, default=100,
                        help="每张原图合成多少张训练图（默认 100）")
    parser.add_argument("--val-split", type=float, default=0.15,
                        help="验证集比例（默认 0.15）")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # 检查是否已有数据集
    if IMG_TRAIN.exists() and IMG_VAL.exists():
        train_count = len(list(IMG_TRAIN.glob("*.jpg")))
        val_count = len(list(IMG_VAL.glob("*.jpg")))
        if train_count > 0 and val_count > 0:
            print(f"[SKIP] 数据集已存在 ({train_count} 训练图, {val_count} 验证图)，跳过生成")
            yaml_path = OUT_DIR / "tarot.yaml"
            if yaml_path.exists():
                print(f"数据集配置: {yaml_path}")
            return

    random.seed(args.seed)
    np.random.seed(args.seed)

    # 扫描原图
    src_files = sorted(SRC_DIR.glob("sm_RWSa-[TWSCP]-*.webp"))  # 过滤掉 X- 系列
    if not src_files:
        print(f"[ERROR] 未找到原图：{SRC_DIR}")
        return

    print(f"找到 {len(src_files)} 张原图")

    # 分配 train / val
    random.shuffle(src_files)
    val_size = int(len(src_files) * args.val_split)
    val_files = src_files[:val_size]
    train_files = src_files[val_size:]

    print(f"训练集: {len(train_files)} 类, 验证集: {len(val_files)} 类")
    print(f"将生成 ~{len(train_files) * args.augment} 张训练图, ~{len(val_files) * args.augment} 张验证图")

    IMG_TRAIN.mkdir(parents=True, exist_ok=True)
    IMG_VAL.mkdir(parents=True, exist_ok=True)
    LBL_TRAIN.mkdir(parents=True, exist_ok=True)
    LBL_VAL.mkdir(parents=True, exist_ok=True)

    total = 0
    for split_files, img_dir, lbl_dir in [
        (train_files, IMG_TRAIN, LBL_TRAIN),
        (val_files,   IMG_VAL,   LBL_VAL),
    ]:
        for card_path in split_files:
            class_id, _, _ = parse_card_info(card_path.name)
            for i in range(args.augment):
                synth_one(card_path, class_id, i, img_dir, lbl_dir)
                total += 1
        print(f"  完成 {img_dir.parent.name}/{img_dir.name}: {len(split_files) * args.augment} 张")

    print(f"\n总计生成 {total} 张合成图 ✓")
    print(f"训练集: {IMG_TRAIN}  标签: {LBL_TRAIN}")
    print(f"验证集: {IMG_VAL}    标签: {LBL_VAL}")

    # 生成 tarot.yaml
    yaml_path = OUT_DIR / "tarot.yaml"
    yaml_content = f"""# Tarot Card Detection Dataset
# 自动生成 by prepare_data.py

path: {OUT_DIR.resolve()}
train: images/train
val: images/val

nc: 78
{build_class_yaml()}
"""
    yaml_path.write_text(yaml_content, encoding="utf-8")
    print(f"\n生成数据集配置: {yaml_path}")


if __name__ == "__main__":
    main()
