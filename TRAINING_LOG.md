# 训练日志模板

## 实验配置

- **模型**: YOLOv8n-seg
- **数据集**: 南极动物数据集
- **GPU**: RTX 4060 (8GB)
- **训练参数**:
  - Epochs: 100
  - Batch Size: 8
  - Image Size: 640
  - Optimizer: AdamW
  - Learning Rate: 0.001

## 训练记录

### Epoch 1-10
- Loss: xxx
- mIoU: xxx
- 备注: 初始训练阶段

### Epoch 11-50
- Loss: xxx
- mIoU: xxx
- 备注: 模型收敛中

### Epoch 51-100
- Loss: xxx
- mIoU: xxx
- 备注: 模型稳定

## 最终结果

- **最佳 mIoU**: xxx (Epoch xx)
- **训练时长**: xxx 小时
- **最佳模型**: runs/segment/antarctic_yolo/weights/best.pt

## 测试结果

- mIoU: xxx
- Dice: xxx
- Pixel Accuracy: xxx
- Precision: xxx
- Recall: xxx
- F1-Score: xxx
- Boundary F1: xxx

## 问题与改进

1. 问题: xxx
   - 解决方案: xxx

2. 改进方向: xxx

## LLM 使用记录

本实验使用了 GitHub Copilot 辅助:
- 代码生成: xxx
- 参数调优建议: xxx
- 调试帮助: xxx
