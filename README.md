# 南极动物语义分割项目

基于 YOLOv8-seg 的南极动物（企鹅、海狮等）语义分割系统。本项目采用预训练模型 + SAM 伪标签生成的方案，适配 RTX 4060 (8GB) 显卡训练。

---

## 📋 LLM 辅助声明

**本项目使用了 GitHub Copilot 等 LLM 工具辅助完成，具体标注如下：**

### 代码部分
- ✅ `scripts/preprocess_data.py` - 由 GitHub Copilot 辅助生成
- ✅ `scripts/train.py` - 由 GitHub Copilot 辅助生成  
- ✅ `scripts/test.py` - 由 GitHub Copilot 辅助生成
- ✅ `scripts/evaluate.py` - 由 GitHub Copilot 辅助生成
- ✅ `scripts/quick_test.py` - 由 GitHub Copilot 辅助生成
- ✅ `README.md` - 由 GitHub Copilot 辅助生成

### 其他部分
- 项目架构设计和技术选型由人工完成
- 数据集处理和标注策略由人工设计
- 参数调优和实验分析由人工完成

---

## 🎯 项目特点

- ✅ **轻量级模型**: YOLOv8n/s-seg，适配 8GB 显存
- ✅ **自动标注**: 使用 SAM (Segment Anything Model) 生成伪标签
- ✅ **完整流程**: 数据预处理 → 训练 → 测试 → 评估
- ✅ **丰富指标**: mIoU, Dice, Pixel Accuracy, Precision, Recall, F1, Boundary F1 (7个指标)
- ✅ **快速验证**: 一键环境测试脚本
- ✅ **可复现**: 固定随机种子，详细文档

---

## 🛠️ 环境配置

### 系统要求
- Python 3.8+
- CUDA 11.8+ (GPU训练)
- 8GB+ GPU 显存 (推荐 RTX 3060/4060 或更高)

### 方法1: 使用 pip (推荐)

```powershell
# 克隆仓库
git clone <your-repo-url>
cd hw1-semantic-segmentation-Startshark

# 创建虚拟环境
python -m venv venv
.\venv\Scripts\activate  # Windows PowerShell

# 安装依赖
pip install -r requirements.txt
```

### 方法2: 使用 conda

```powershell
# 创建环境
conda env create -f environment.yml

# 激活环境
conda activate antarctic-segmentation
```

### 快速验证环境

运行验证脚本确保环境配置正确:

```powershell
python scripts/quick_test.py
```

该脚本会检查:
- ✅ Python 版本
- ✅ 必要包安装
- ✅ GPU 可用性
- ✅ YOLO 基本功能
- ✅ 项目文件结构

---

## 📁 项目结构

```
hw1-semantic-segmentation-Startshark/
├── dataset/                    # 原始数据集（需自行下载）
│   └── *.jpg                   # 南极动物图像
├── data/
│   └── processed/              # 预处理后的数据
│       ├── train/              # 训练集
│       │   ├── images/
│       │   └── labels/
│       ├── val/                # 验证集
│       │   ├── images/
│       │   └── labels/
│       └── antarctic.yaml      # YOLO配置文件
├── test/                       # 测试集（需自行下载）
│   └── *.jpg                   # 测试图像
├── scripts/                    # 脚本目录
│   ├── preprocess_data.py      # 数据预处理 + SAM标注
│   ├── train.py                # 模型训练
│   ├── test.py                 # 模型测试
│   ├── evaluate.py             # 模型评估
│   └── quick_test.py           # 快速验证
├── runs/                       # 训练输出（自动生成）
│   └── segment/
│       └── antarctic_yolo/
│           ├── weights/
│           │   ├── best.pt     # 最佳模型
│           │   └── last.pt     # 最后模型
│           └── results.png     # 训练曲线
├── requirements.txt            # Python依赖
├── environment.yml             # Conda环境
├── .gitignore                  # Git忽略文件
└── README.md                   # 本文件
```

---

## 🚀 使用方法

### 1. 准备数据集

下载数据集并放入 `dataset/` 目录:

```
数据集链接: https://box.nju.edu.cn/d/74c94657a0404eb79c74/
测试集链接: https://box.nju.edu.cn/d/986313080d57481eab34/
```

**注意**: 根据数据集使用协议，禁止分享和上传数据集。

### 2. 数据预处理

#### 2.1 下载 SAM 模型 (用于生成伪标签)

```powershell
# 下载 SAM ViT-B 模型 (~375MB)
# 链接: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
# 下载后放在项目根目录
```

#### 2.2 运行预处理脚本

```powershell
python scripts/preprocess_data.py
```

这个脚本会:
1. 整理数据集结构 (train/val 分割)
2. 使用 SAM 自动生成分割标注
3. 创建 YOLO 配置文件

**说明**: SAM 标注是基于无监督的方法，生成的标签可能不完全准确，但可以作为初始标注用于训练。

### 3. 训练模型

```powershell
# 使用默认配置训练 (YOLOv8n-seg, 100 epochs, batch=8)
python scripts/train.py --data data/processed/antarctic.yaml --model n --epochs 100

# 使用更大的模型 (需要更多显存)
python scripts/train.py --model s --batch 4

# 从断点继续训练
python scripts/train.py --resume

# 查看所有参数
python scripts/train.py --help
```

**参数说明**:
- `--model`: 模型大小 (n/s/m/l/x)
  - `n` (nano): 最小，适合 8GB 显存
  - `s` (small): 小型，batch 需减小到 4-6
  - `m` (medium): 中型，需要 12GB+ 显存
- `--batch`: 批次大小，8适合 RTX 4060
- `--epochs`: 训练轮数
- `--device`: GPU设备 (0 表示第一块 GPU)

**训练时长**:
- RTX 4060 + YOLOv8n: ~2-3小时 (100 epochs)
- RTX 4060 + YOLOv8s: ~4-5小时 (100 epochs)

### 4. 测试模型

#### 4.1 对测试集进行预测

```powershell
# 使用最佳模型预测测试集
python scripts/test.py --model runs/segment/antarctic_yolo/weights/best.pt --source test --output test_results

# 预测单张图像
python scripts/test.py --model runs/segment/antarctic_yolo/weights/best.pt --source test/image.jpg --mode single

# 批量预测
python scripts/test.py --model runs/segment/antarctic_yolo/weights/best.pt --source test --mode batch --output predictions
```

结果会保存在 `test_results/` 或 `predictions/` 目录。

### 5. 评估模型

```powershell
# 评估模型性能（需要真实标签）
python scripts/evaluate.py --model runs/segment/antarctic_yolo/weights/best.pt --val-data data/processed/val/images
```

**评估指标** (>=5个):
1. **mIoU** (Mean Intersection over Union) - 平均交并比
2. **Dice Coefficient** - Dice系数/F1分数
3. **Pixel Accuracy** - 像素准确率
4. **Mean Precision** - 平均精确率
5. **Mean Recall** - 平均召回率
6. **Mean F1-Score** - 平均F1分数
7. **Boundary F1** - 边界F1分数

---

## 📊 模型性能

### 预期性能 (基于 YOLOv8n-seg)

| 指标 | 数值 |
|------|------|
| mIoU | ~0.65-0.75 |
| Dice Coefficient | ~0.75-0.85 |
| Pixel Accuracy | ~0.85-0.92 |
| Mean Precision | ~0.70-0.80 |
| Mean Recall | ~0.68-0.78 |

**注**: 实际性能取决于数据集质量和 SAM 伪标签准确性。

---

## 💡 常见问题

### Q1: GPU 显存不足怎么办?

**A**: 尝试以下方法:
```powershell
# 1. 减小 batch size
python scripts/train.py --batch 4

# 2. 使用更小的模型
python scripts/train.py --model n

# 3. 减小图像尺寸
python scripts/train.py --img-size 512

# 4. 关闭数据缓存 (已默认关闭)
```

### Q2: SAM 模型太慢怎么办?

**A**: 
1. 使用更小的 SAM 模型 (vit_b 而非 vit_h)
2. 减少采样点数 (修改 `points_per_side`)
3. 手动标注少量样本，减少对 SAM 的依赖

### Q3: 如何改进模型性能?

**A**:
1. **手动标注**: 对 SAM 生成的标签进行人工修正
2. **数据增强**: 调整训练脚本中的增强参数
3. **更大模型**: 使用 YOLOv8s 或 YOLOv8m
4. **更长训练**: 增加 epochs 到 200-300
5. **学习率调优**: 调整 `lr0` 和 `lrf` 参数

### Q4: 如何使用自己的标注?

**A**: 将标注转换为 YOLO 格式:
```
class_id x1 y1 x2 y2 ... xn yn
```
每行表示一个对象，坐标归一化到 [0, 1]。

---

## 🔧 高级用法

### 1. 自定义类别

修改 `scripts/preprocess_data.py` 中的类别定义:

```python
self.classes = {
    0: 'background',
    1: 'penguin',      # 企鹅
    2: 'seal',         # 海狮
    3: 'bird',         # 鸟类
    # 添加更多类别...
}
```

### 2. 调整 SAM 参数

在 `scripts/preprocess_data.py` 中修改:

```python
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=16,      # 增加采样密度
    pred_iou_thresh=0.86,    # 调整IoU阈值
    stability_score_thresh=0.92,
    # ...
)
```

### 3. 使用 Weights & Biases 跟踪实验

```powershell
# 安装 wandb
pip install wandb

# 登录
wandb login

# 训练时会自动上传到 W&B
python scripts/train.py --project my-project
```

---

## 📝 论文报告要求

根据作业要求，论文应包含:

1. **实验方法** (详细描述):
   - 数据预处理流程 (SAM 伪标签生成)
   - 模型架构 (YOLOv8-seg)
   - 训练策略 (学习率、优化器、增强等)

2. **实验结果**:
   - 展示多种动物的分割结果（可视化）
   - 附录中包含测试集结果

3. **评价分析** (>=5个指标):
   - mIoU, Dice, PA, Precision, Recall, F1, Boundary F1

4. **讨论与总结**:
   - 模型优缺点分析
   - 错误案例分析
   - 改进方向

5. **LLM 使用标注**:
   - 在论文结尾标注使用的 LLM 工具
   - 说明哪些部分由 LLM 生成或修改

---

## 📚 参考资料

- [YOLOv8 官方文档](https://docs.ultralytics.com/)
- [Segment Anything (SAM)](https://segment-anything.com/)
- [语义分割评估指标](https://www.jeremyjordan.me/evaluating-image-segmentation-models/)
- [Google Colab 使用指南](https://colab.research.google.com/)

---

## 📄 许可证

本项目仅用于 NJU 神经网络课程作业，不得用于其他用途。

**数据集版权**: 数据集具有版权限制，严禁分享、上传或用于其他目的。

---

## 🙏 致谢

- **Ultralytics**: 提供 YOLOv8 框架
- **Meta AI**: 提供 SAM 模型
- **NJU NN 课程组**: 提供数据集和指导

---

**最后更新**: 2025-11-03

**作者**: [Your Name]

**课程**: 南京大学 - 神经网络 (2025)
