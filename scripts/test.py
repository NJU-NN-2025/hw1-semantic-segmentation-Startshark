import os
import cv2
import torch
from ultralytics import YOLO
from pathlib import Path
import argparse
import numpy as np
from tqdm import tqdm
import json


def test_model(
    model_path: str,
    source: str = "test",
    output_dir: str = "test_results",
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.7,
    save_txt: bool = True,
    save_conf: bool = True,
    save_img: bool = True,
):
    """
    使用训练好的模型进行测试
    
    Args:
        model_path: 模型权重路径
        source: 测试图像路径（文件夹或单个文件）
        output_dir: 输出目录
        conf_threshold: 置信度阈值
        iou_threshold: IoU阈值
        save_txt: 是否保存标注文件
        save_conf: 是否保存置信度
        save_img: 是否保存可视化图像
    """
    
    print("=" * 60)
    print("南极动物语义分割 - 模型测试")
    print("=" * 60)
    
    # 检查模型文件
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    # 加载模型
    print(f"\n加载模型: {model_path}")
    model = YOLO(model_path)
    
    # 检查设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 进行预测
    print(f"\n开始预测: {source}")
    print("=" * 60)
    
    results = model.predict(
        source=source,
        conf=conf_threshold,
        iou=iou_threshold,
        save=save_img,
        save_txt=save_txt,
        save_conf=save_conf,
        project=str(output_path),
        name="predictions",
        device=device,
        verbose=True,
    )
    
    print("\n" + "=" * 60)
    print(f"预测完成! 结果保存在: {output_path / 'predictions'}")
    print("=" * 60)
    
    return results


def predict_single_image(
    model_path: str,
    image_path: str,
    output_path: str = None,
    conf_threshold: float = 0.25,
    show_labels: bool = True,
    show_conf: bool = True,
):
    """
    对单张图像进行预测并可视化
    
    Args:
        model_path: 模型权重路径
        image_path: 输入图像路径
        output_path: 输出图像路径（可选）
        conf_threshold: 置信度阈值
        show_labels: 是否显示类别标签
        show_conf: 是否显示置信度
    """
    
    # 加载模型
    model = YOLO(model_path)
    
    # 预测
    results = model.predict(
        source=image_path,
        conf=conf_threshold,
        save=False,
        verbose=False,
    )
    
    # 获取第一个结果
    result = results[0]
    
    # 可视化
    annotated_img = result.plot(
        conf=show_conf,
        labels=show_labels,
        boxes=True,
        masks=True,
    )
    
    # 保存或显示
    if output_path:
        cv2.imwrite(output_path, annotated_img)
        print(f"结果已保存: {output_path}")
    else:
        cv2.imshow("Prediction", annotated_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return result


def batch_predict(
    model_path: str,
    image_dir: str,
    output_dir: str = "predictions",
    conf_threshold: float = 0.25,
):
    """
    批量预测图像
    
    Args:
        model_path: 模型权重路径
        image_dir: 图像目录
        output_dir: 输出目录
        conf_threshold: 置信度阈值
    """
    
    print("=" * 60)
    print("批量预测")
    print("=" * 60)
    
    # 加载模型
    model = YOLO(model_path)
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 获取所有图像
    image_dir = Path(image_dir)
    image_files = list(image_dir.glob("*.jpg")) + \
                 list(image_dir.glob("*.jpeg")) + \
                 list(image_dir.glob("*.png"))
    
    print(f"找到 {len(image_files)} 张图像")
    
    # 预测结果统计
    stats = {
        'total': len(image_files),
        'processed': 0,
        'failed': 0,
    }
    
    # 逐个预测
    for img_path in tqdm(image_files, desc="预测进度"):
        try:
            # 预测
            results = model.predict(
                source=str(img_path),
                conf=conf_threshold,
                save=False,
                verbose=False,
            )
            
            # 可视化并保存
            result = results[0]
            annotated_img = result.plot()
            
            output_file = output_path / img_path.name
            cv2.imwrite(str(output_file), annotated_img)
            
            stats['processed'] += 1
            
        except Exception as e:
            print(f"\n处理 {img_path.name} 时出错: {e}")
            stats['failed'] += 1
    
    # 打印统计
    print("\n" + "=" * 60)
    print("批量预测完成!")
    print(f"  总数: {stats['total']}")
    print(f"  成功: {stats['processed']}")
    print(f"  失败: {stats['failed']}")
    print(f"  结果保存在: {output_path}")
    print("=" * 60)


def main():
    """命令行接口"""
    parser = argparse.ArgumentParser(description="YOLO模型测试和推理")
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="模型权重路径 (.pt文件)"
    )
    
    parser.add_argument(
        "--source",
        type=str,
        default="test",
        help="测试图像路径（文件或文件夹）"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="test_results",
        help="输出目录"
    )
    
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="置信度阈值"
    )
    
    parser.add_argument(
        "--iou",
        type=float,
        default=0.7,
        help="IoU阈值"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        default="test",
        choices=["test", "single", "batch"],
        help="运行模式"
    )
    
    args = parser.parse_args()
    
    if args.mode == "test":
        # 标准测试模式
        test_model(
            model_path=args.model,
            source=args.source,
            output_dir=args.output,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
        )
    
    elif args.mode == "single":
        # 单图预测
        predict_single_image(
            model_path=args.model,
            image_path=args.source,
            conf_threshold=args.conf,
        )
    
    elif args.mode == "batch":
        # 批量预测
        batch_predict(
            model_path=args.model,
            image_dir=args.source,
            output_dir=args.output,
            conf_threshold=args.conf,
        )


if __name__ == "__main__":
    main()
