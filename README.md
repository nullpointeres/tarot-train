# Tarot Card Detector — 训练项目

基于 78 张塔罗牌原图，通过合成数据训练 YOLOv11 目标检测模型。

## 环境要求

- Python 3.9+
- NVIDIA GPU（推荐，训练约 2-4 小时）；Mac M 系列可用 `mps`；CPU 可跑但较慢

## 快速开始

### 一键训练（推荐）

```bash
cd tarot-train
chmod +x train.sh
./train.sh
```

### 分步运行

```bash
# 1. 安装依赖
pip install ultralytics pillow numpy

# 2. 生成合成数据集（每张原图合成 100 张，共约 7800 张训练图）
python scripts/prepare_data.py --augment 100

# 3. 训练模型（200 轮，预计 GPU 2-4 小时）
python scripts/train.py --epochs 200 --device 0

# 4. 测试推理
python scripts/predict.py --conf 0.5
```

## 脚本说明

| 脚本 | 作用 |
|------|------|
| `scripts/prepare_data.py` | 扫描 tarot-static 原图，合成多样化训练数据（背景/角度/亮度变化）|
| `scripts/train.py` | YOLO 训练脚本，含自动设备检测、早停、ONNX 导出 |
| `scripts/predict.py` | 在任意图片上测试模型，输出检出的牌名 |
| `train.sh` | 一键入口脚本 |

## 数据集配置

自动生成的 `dataset/tarot.yaml` 定义了 78 类（card_no 0-77）：

| ID | 类别 |
|----|------|
| 0-21 | 大阿卡纳（愚人 → 世界） |
| 22-35 | 权杖（Ace → King） |
| 36-49 | 圣杯（Ace → King） |
| 50-63 | 宝剑（Ace → King） |
| 64-77 | 星币（Ace → King） |

## 训练参数建议

| 场景 | 建议 |
|------|------|
| 数据量少（78 张原图） | augment=200+，epochs=300 |
| 有 GPU 显存限制 | batch=8 |
| Mac M 系列 | `--device mps` |
| 想快速验证 | epochs=50，augment=50 |

## 输出

```
tarot-train/
├── runs/train/weights/best.pt   ← 最佳模型（用于推理）
├── runs/train/weights/last.pt   ← 最后模型（可继续训练）
└── models/best.onnx             ← ONNX 格式（用于部署）
```
