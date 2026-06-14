# 乳腺超声图像智能诊断系统

**生物医学工程竞赛项目** | 多任务深度学习 + 稀疏加权 Logit 融合

---

## 项目概览

- **任务**: 乳腺超声图像良恶性分类（分割辅助）
- **数据**: 1,875 张乳腺超声图像，患者级划分
- **最佳方案**: V10 分割→分类联合推理 (V9 DISTILL+SOUP+V2.3 + MiT-B2 分割 ROI)
- **最佳性能**: **AUC = 0.9225**, best F1 = 0.7981 @0.545
- **35 个实验** 完整记录

---

## 项目结构

```
生物医学工程竞赛/
│
├── src/                              # 核心源码包
│   ├── config.py                     # 统一配置 + 模型注册表
│   ├── models/                       # 模型定义
│   │   ├── base.py                   # BreastCancerMultiTaskNet (V2.3)
│   │   ├── v5.py                     # V5Model, V5ModelUNetPP
│   │   └── v6.py                     # BreastCancerMultiTaskNetV6
│   ├── data/                         # 数据管线
│   │   ├── dataset.py                # BreastUltrasoundDataset
│   │   └── split.py                  # 患者级 train/val 划分
│   ├── losses/                       # 损失函数
│   │   ├── focal.py                  # FocalLoss
│   │   └── lomix.py                  # LoMix 组合损失
│   ├── training/                     # 训练脚本
│   │   ├── train_v5.py               # V5.1/V5.2
│   │   ├── train_v6.py               # V6 LoMix
│   │   ├── train_v7.py               # V7.4/V7.6 SAM
│   │   └── train_distill.py          # 知识蒸馏
│   ├── inference/                    # 推理引擎
│   │   └── engine.py                 # 模型加载 / TTA 融合 / V10 分割→分类
│   ├── segmentation/                 # 分割管线 (V10)
│   │   └── roi_pipeline.py           # MiT-B2 UNet 两阶段粗+精分割
│   └── evaluation/                   # 评估脚本
│       ├── test_single.py            # 单模型测试
│       ├── test_ensemble.py          # 多模型融合测试
│       └── seg_then_classify.py      # 分割→分类联合评估 (V10)
│
├── app/                              # Streamlit Web 应用
│   ├── main.py                       # 三页 UI (Apple×IBM 设计系统)
│   └── report.py                     # 临床报告 HTML + PDF 生成
│
├── scripts/                          # 研究工具
│   ├── eda.py                        # 探索性数据分析
│   ├── generate_masks.py             # 批量生成分割掩膜
│   ├── make_model_soup.py            # 模型权重汤
│   └── search_fusion_weights.py      # 融合权重搜索
│
├── weights/                          # 模型权重 (.gitignore)
├── outputs/                          # 运行时输出 (.gitignore)
├── data/                             # 数据集
├── experiments/                      # 历史实验归档 (V2-V8)
├── lomix/                            # LoMix 论文参考
├── docs/                             # 项目文档
│   ├── 运行结果.txt
│   └── 赛题.pdf
│
├── requirements.txt
└── .gitignore
```

---

## 性能演进

| 阶段 | 方案 | AUC | 关键突破 |
|------|------|-----|----------|
| V1 | Baseline (样本级划分) | 0.905 | 数据泄漏警告 |
| V2.3 | 患者级划分 + 固定 Resize | **0.903** | 诚实基线 (22 实验未超越) |
| V5 | 3 模型异质融合 | **0.907** | 首次突破 0.903 墙 |
| V7 | 4 模型融合 + SAM | **0.909** | 异构融合历史记录 |
| V9 | 3 模型加权 logit 融合 | **0.914** | 纯分类最佳 |
| V10 | 分割→分类联合推理 | **0.9225** | 当前最佳 (外部 0.9232) |

---

## 快速开始

### 环境要求

```bash
pip install -r requirements.txt
```

主要依赖: PyTorch >= 2.0, segmentation-models-pytorch, albumentations, streamlit, opencv-python

PDF 生成依赖 weasyprint，Windows 下通过 conda 安装可自动处理 GTK 系统库：

```bash
conda install -c conda-forge weasyprint -y
```

> Linux/macOS: `pip install weasyprint` 即可。

### 启动 Web 应用

```bash
cd 生物医学工程竞赛
streamlit run app/main.py
```

浏览器打开 `http://localhost:8501`，三个页面：

| 页面 | 功能 |
|------|------|
| **单病例智能工作站** | 上传超声图 → AI 推理 → 查看诊断结果 → **一键生成并下载 PDF 报告** |
| **高通量批量筛查中心** | 批量上传 → 自动筛查 → 统计总览 + CSV 导出 |
| **算法技术白皮书** | EDA 洞察 / 实验演进时间轴 / 性能对比 / 技术栈 / 核心经验 |

> 设计系统: Apple × IBM 融合 — IBM Plex Sans 字体、IBM Blue 单强调色、零渐变、4px 栅格。

### 训练

```bash
# V5.1 BI-RADS 多任务
python -m src.training.train_v5 --variant 1

# V6 LoMix
python -m src.training.train_v6

# V7.6 SAM 优化器
python -m src.training.train_v7 --variant 6 --sam

# 知识蒸馏
python -m src.training.train_distill --student-seed 42
```

### 测试

```bash
# 单模型测试
python -m src.evaluation.test_single

# V9 加权 logit 融合
python src/evaluation/test_ensemble.py --models DISTILL,SOUP,V2.3 --device cuda --fusion logit_mean --weights 0.043422758938255444,0.8786080776229839,0.07796916343876055

# 两模型 compact
python src/evaluation/test_ensemble.py --models DISTILL,SOUP --device cuda --fusion prob_mean

# V10 分割→分类联合评估 (需先生成分割掩膜)
python -m src.evaluation.seg_then_classify \
    --image-dir data/测试集 \
    --mask-dir outputs/predicted_masks/test \
    --output-dir outputs/seg_then_classify/test

# 批量分割推理
python -m src.segmentation.roi_pipeline \
    --input-dir data/测试集/benign \
    --output-dir outputs/predicted_masks/test/benign \
    --coarse weights/seg_unet_mit_b2_fold0_s384_seed3101_best.pth weights/seg_unet_mit_b2_all_s384_seed6201_epoch12.pth \
    --roi weights/seg_roi_unet_mit_b2_all_best.pth
```

---

## 核心技术栈

| 组件 | 选型 |
|------|------|
| 骨干网络 | ResNet34 (ImageNet 预训练) |
| 分割架构 | U-Net / UNet++ |
| 分类损失 | FocalLoss (α=0.7, γ=2) |
| 分割损失 | BCEWithLogitsLoss |
| 优化器 | AdamW + SAM |
| 数据增强 | 固定 Resize(256) + HFlip + Affine + GaussNoise |
| TTA | 水平翻转（单尺度） |
| 混合精度 | AMP (torch.amp) |
| Web UI | Streamlit + Apple×IBM 设计系统 |

---

## 关键经验

- ✅ **患者级划分**是必须的，样本级划分导致 AUC 虚高
- ✅ **异质融合**是突破单模型天花板的唯一有效方法
- ✅ **概率平均**优于 Stacking/加权（小验证集过拟合）
- ❌ 更大架构（DenseNet/EfficientNet）在小数据上过拟合
- ❌ 多尺度 TTA / 多尺度架构均有害于 1507 样本场景

---

<p align="center">
  <sub>© 2026 生物医学工程竞赛团队 | 35 实验 · V10 分割→分类 · AUC 0.9225</sub>
</p>
