"""
快速开始示例 - 完整工作流程演示
LLM 辅助: 本文件由 GitHub Copilot 辅助生成
"""

import os
from pathlib import Path


def print_step(step_num: int, title: str):
    """打印步骤标题"""
    print("\n" + "=" * 60)
    print(f"步骤 {step_num}: {title}")
    print("=" * 60)


def main():
    """快速开始演示"""
    
    print("=" * 60)
    print("南极动物语义分割 - 快速开始指南")
    print("=" * 60)
    print("\n本脚本将引导你完成完整的工作流程")
    
    # 步骤1: 环境验证
    print_step(1, "环境验证")
    print("运行环境验证测试...")
    print("\n命令:")
    print("  python scripts/quick_test.py")
    print("\n请先运行上述命令验证环境!")
    
    response = input("\n环境验证通过了吗? (y/n): ").lower()
    if response != 'y':
        print("\n请先完成环境配置:")
        print("  pip install -r requirements.txt")
        return
    
    # 步骤2: 数据集准备
    print_step(2, "数据集准备")
    print("下载数据集:")
    print("  训练集: https://box.nju.edu.cn/d/74c94657a0404eb79c74/")
    print("  测试集: https://box.nju.edu.cn/d/986313080d57481eab34/")
    print("\n将图像放入以下目录:")
    print("  dataset/  (训练集)")
    print("  test/     (测试集)")
    
    # 检查数据集
    if not Path('dataset').exists() or not list(Path('dataset').glob('*.jpg')):
        print("\n⚠ 未检测到数据集，请先下载并放入 dataset/ 目录")
        response = input("\n继续演示 (仅显示命令)? (y/n): ").lower()
        if response != 'y':
            return
    
    # 步骤3: SAM 模型下载
    print_step(3, "SAM 模型下载 (可选)")
    print("如果需要使用 SAM 自动生成标注:")
    print("  下载链接: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth")
    print("  大小: ~375MB")
    print("  放置位置: 项目根目录")
    print("\n如果已有标注数据，可跳过此步骤")
    
    # 步骤4: 数据预处理
    print_step(4, "数据预处理")
    print("运行预处理脚本:")
    print("\n命令:")
    print("  python scripts/preprocess_data.py")
    print("\n这将:")
    print("  1. 整理数据集结构 (train/val 分割)")
    print("  2. (可选) 使用 SAM 生成伪标签")
    print("  3. 创建 YOLO 配置文件")
    
    # 步骤5: 模型训练
    print_step(5, "模型训练")
    print("开始训练模型:")
    print("\n基础命令:")
    print("  python scripts/train.py --data data/processed/antarctic.yaml --model n --epochs 100")
    print("\n参数说明:")
    print("  --model n      # 使用 nano 模型 (适合 8GB 显存)")
    print("  --epochs 100   # 训练 100 轮")
    print("  --batch 8      # 批次大小 8")
    print("\n高级选项:")
    print("  --model s      # 使用 small 模型 (更好性能,需要调整 batch)")
    print("  --resume       # 从断点继续训练")
    print("\n预计训练时间 (RTX 4060):")
    print("  YOLOv8n: ~2-3 小时")
    print("  YOLOv8s: ~4-5 小时")
    
    # 步骤6: 模型测试
    print_step(6, "模型测试")
    print("使用训练好的模型进行预测:")
    print("\n命令:")
    print("  python scripts/test.py --model runs/segment/antarctic_yolo/weights/best.pt --source test")
    print("\n单张图像预测:")
    print("  python scripts/test.py --model runs/segment/antarctic_yolo/weights/best.pt --source test/image.jpg --mode single")
    print("\n结果将保存在 test_results/ 目录")
    
    # 步骤7: 模型评估
    print_step(7, "模型评估")
    print("评估模型性能 (需要真实标签):")
    print("\n命令:")
    print("  python scripts/evaluate.py --model runs/segment/antarctic_yolo/weights/best.pt --val-data data/processed/val/images")
    print("\n将计算以下指标:")
    print("  1. mIoU (Mean IoU)")
    print("  2. Dice Coefficient")
    print("  3. Pixel Accuracy")
    print("  4. Mean Precision")
    print("  5. Mean Recall")
    print("  6. Mean F1-Score")
    print("  7. Boundary F1")
    
    # 步骤8: 结果整理
    print_step(8, "结果整理")
    print("为论文准备结果:")
    print("\n1. 可视化结果:")
    print("   - 查看 test_results/ 目录中的预测图像")
    print("   - 选择最佳案例放入论文附录")
    print("\n2. 评估指标:")
    print("   - 使用 evaluation_results.json 中的指标")
    print("   - 制作表格和图表")
    print("\n3. 训练曲线:")
    print("   - 查看 runs/segment/antarctic_yolo/results.png")
    print("   - 分析训练过程")
    
    # 总结
    print("\n" + "=" * 60)
    print("快速开始指南完成!")
    print("=" * 60)
    print("\n完整流程:")
    print("  1. ✅ 环境验证: python scripts/quick_test.py")
    print("  2. 📁 数据集下载并放入 dataset/ 和 test/")
    print("  3. 🔄 数据预处理: python scripts/preprocess_data.py")
    print("  4. 🏋️ 模型训练: python scripts/train.py --data data/processed/antarctic.yaml")
    print("  5. 🧪 模型测试: python scripts/test.py --model <path> --source test")
    print("  6. 📊 模型评估: python scripts/evaluate.py --model <path> --val-data <path>")
    print("  7. 📝 整理论文报告")
    
    print("\n提示:")
    print("  - 所有脚本都支持 --help 查看详细参数")
    print("  - 训练过程会自动保存检查点")
    print("  - 建议使用 GPU 训练以节省时间")
    print("  - 详细文档请查看 README.md")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
