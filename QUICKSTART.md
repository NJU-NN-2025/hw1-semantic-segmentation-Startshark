# 快速使用指南

## 第一次使用

### 1. 克隆仓库

```powershell
git clone <your-repo-url>
cd hw1-semantic-segmentation-Startshark
```

### 2. 安装依赖

```powershell
pip install -r requirements.txt
```

### 3. 验证环境

```powershell
python scripts/quick_test.py
```

### 4. 下载数据集

- 训练集: https://box.nju.edu.cn/d/74c94657a0404eb79c74/
- 测试集: https://box.nju.edu.cn/d/986313080d57481eab34/

放置位置:
```
dataset/     # 训练集图像
test/        # 测试集图像
```

### 5. (可选) 下载 SAM 模型

如需自动生成标注:
```
https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

## 基本使用流程

### 方式1: 一键运行 (推荐)

**PowerShell:**
```powershell
.\run_all.ps1
```

**或 CMD:**
```cmd
run_all.bat
```

### 方式2: 手动步骤

#### 步骤1: 数据预处理
```powershell
python scripts/preprocess_data.py
```

#### 步骤2: 训练模型
```powershell
# 基础训练 (YOLOv8n, 100轮)
python scripts/train.py --data data/processed/antarctic.yaml

# 使用更大模型
python scripts/train.py --model s --batch 4
```

#### 步骤3: 测试模型
```powershell
python scripts/test.py --model runs/segment/antarctic_yolo/weights/best.pt --source test
```

#### 步骤4: 评估性能
```powershell
python scripts/evaluate.py --model runs/segment/antarctic_yolo/weights/best.pt --val-data data/processed/val/images
```

## 常用命令

### 查看帮助
```powershell
python scripts/train.py --help
python scripts/test.py --help
python scripts/evaluate.py --help
```

### 从断点继续训练
```powershell
python scripts/train.py --resume
```

### 预测单张图像
```powershell
python scripts/test.py --model <model.pt> --source image.jpg --mode single
```

### 批量预测
```powershell
python scripts/test.py --model <model.pt> --source test --mode batch
```

## 文件位置

### 输入
- 原始数据: `dataset/`
- 测试集: `test/`
- SAM 模型: `sam_vit_b_01ec64.pth` (根目录)

### 输出
- 训练结果: `runs/segment/antarctic_yolo/`
- 模型权重: `runs/segment/antarctic_yolo/weights/best.pt`
- 测试结果: `test_results/predictions/`
- 评估结果: `evaluation_results.json`

## 故障排除

### 环境问题
```powershell
# 重新安装依赖
pip install --upgrade -r requirements.txt

# 验证环境
python scripts/quick_test.py
```

### 显存不足
```powershell
# 减小 batch size
python scripts/train.py --batch 4

# 使用更小模型
python scripts/train.py --model n
```

### 找不到模型
- 确保训练完成: 检查 `runs/segment/antarctic_yolo/weights/`
- 或下载预训练模型: YOLOv8 会自动下载

## 需要帮助?

1. 查看完整文档: `README.md`
2. 运行快速测试: `python scripts/quick_test.py`
3. 查看训练日志: `TRAINING_LOG.md`
4. 查看项目配置: `python config.py`

## LLM 辅助标注

本项目使用了 GitHub Copilot 辅助开发，详见各文件开头注释。
