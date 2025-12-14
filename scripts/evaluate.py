import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from pathlib import Path
import argparse
from tqdm import tqdm
import json
from sklearn.metrics import precision_recall_fscore_support
from scipy.spatial import distance


class SegmentationMetrics:
    """语义分割评估指标计算器"""
    
    def __init__(self, num_classes: int = 3):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        """重置所有指标"""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
        self.total_pixels = 0
        
    def update(self, pred_mask: np.ndarray, gt_mask: np.ndarray):
        """
        更新混淆矩阵
        
        Args:
            pred_mask: 预测mask (H, W) 值为类别ID
            gt_mask: 真实mask (H, W) 值为类别ID
        """
        # 确保形状一致
        assert pred_mask.shape == gt_mask.shape
        
        # 展平
        pred = pred_mask.flatten()
        gt = gt_mask.flatten()
        
        # 更新混淆矩阵
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                self.confusion_matrix[i, j] += np.sum((gt == i) & (pred == j))
        
        self.total_pixels += pred.size
    
    def get_miou(self) -> float:
        """
        计算 Mean Intersection over Union (mIoU)
        这是语义分割最重要的指标
        """
        iou_per_class = []
        
        for i in range(self.num_classes):
            # TP = 对角线元素
            tp = self.confusion_matrix[i, i]
            # FP = 该列其他元素之和
            fp = np.sum(self.confusion_matrix[:, i]) - tp
            # FN = 该行其他元素之和
            fn = np.sum(self.confusion_matrix[i, :]) - tp
            
            # IoU = TP / (TP + FP + FN)
            denominator = tp + fp + fn
            if denominator > 0:
                iou = tp / denominator
                iou_per_class.append(iou)
        
        # 返回平均IoU
        return np.mean(iou_per_class) if iou_per_class else 0.0
    
    def get_dice_coefficient(self) -> float:
        """
        计算 Dice系数 (F1 Score)
        Dice = 2 * TP / (2 * TP + FP + FN)
        """
        dice_per_class = []
        
        for i in range(self.num_classes):
            tp = self.confusion_matrix[i, i]
            fp = np.sum(self.confusion_matrix[:, i]) - tp
            fn = np.sum(self.confusion_matrix[i, :]) - tp
            
            denominator = 2 * tp + fp + fn
            if denominator > 0:
                dice = 2 * tp / denominator
                dice_per_class.append(dice)
        
        return np.mean(dice_per_class) if dice_per_class else 0.0
    
    def get_pixel_accuracy(self) -> float:
        """
        计算 Pixel Accuracy (PA)
        PA = (TP across all classes) / Total Pixels
        """
        correct_pixels = np.trace(self.confusion_matrix)
        return correct_pixels / self.total_pixels if self.total_pixels > 0 else 0.0
    
    def get_class_iou(self) -> dict:
        """
        计算每个类别的IoU
        """
        class_iou = {}
        
        for i in range(self.num_classes):
            tp = self.confusion_matrix[i, i]
            fp = np.sum(self.confusion_matrix[:, i]) - tp
            fn = np.sum(self.confusion_matrix[i, :]) - tp
            
            denominator = tp + fp + fn
            iou = tp / denominator if denominator > 0 else 0.0
            class_iou[f'Class_{i}'] = iou
        
        return class_iou
    
    def get_precision_recall_f1(self) -> dict:
        """
        计算 Precision, Recall, F1-Score
        """
        metrics = {
            'precision': [],
            'recall': [],
            'f1': []
        }
        
        for i in range(self.num_classes):
            tp = self.confusion_matrix[i, i]
            fp = np.sum(self.confusion_matrix[:, i]) - tp
            fn = np.sum(self.confusion_matrix[i, :]) - tp
            
            # Precision = TP / (TP + FP)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            
            # Recall = TP / (TP + FN)
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            # F1 = 2 * Precision * Recall / (Precision + Recall)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            metrics['precision'].append(precision)
            metrics['recall'].append(recall)
            metrics['f1'].append(f1)
        
        return {
            'mean_precision': np.mean(metrics['precision']),
            'mean_recall': np.mean(metrics['recall']),
            'mean_f1': np.mean(metrics['f1']),
            'per_class': metrics
        }
    
    def get_boundary_f1(self, pred_mask: np.ndarray, gt_mask: np.ndarray, 
                       threshold: float = 2.0) -> float:
        """
        计算 Boundary F1-Score
        评估分割边界的准确性
        
        Args:
            pred_mask: 预测mask
            gt_mask: 真实mask  
            threshold: 边界匹配阈值（像素）
        """
        # 提取边界
        pred_edges = self._extract_edges(pred_mask)
        gt_edges = self._extract_edges(gt_mask)
        
        # 计算边界点
        pred_points = np.argwhere(pred_edges > 0)
        gt_points = np.argwhere(gt_edges > 0)
        
        if len(pred_points) == 0 or len(gt_points) == 0:
            return 0.0
        
        # 计算匹配的边界点数量
        matched_pred = 0
        for p in pred_points:
            # 找到最近的真实边界点
            min_dist = np.min(np.linalg.norm(gt_points - p, axis=1))
            if min_dist <= threshold:
                matched_pred += 1
        
        matched_gt = 0
        for g in gt_points:
            # 找到最近的预测边界点
            min_dist = np.min(np.linalg.norm(pred_points - g, axis=1))
            if min_dist <= threshold:
                matched_gt += 1
        
        # 计算 Precision 和 Recall
        precision = matched_pred / len(pred_points) if len(pred_points) > 0 else 0.0
        recall = matched_gt / len(gt_points) if len(gt_points) > 0 else 0.0
        
        # F1-Score
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return f1
    
    def _extract_edges(self, mask: np.ndarray) -> np.ndarray:
        """提取mask边界"""
        # 使用Canny边缘检测
        edges = cv2.Canny(mask.astype(np.uint8), 50, 150)
        return edges
    
    def get_all_metrics(self) -> dict:
        """获取所有评估指标"""
        prf = self.get_precision_recall_f1()
        
        return {
            'mIoU': self.get_miou(),
            'Dice_Coefficient': self.get_dice_coefficient(),
            'Pixel_Accuracy': self.get_pixel_accuracy(),
            'Mean_Precision': prf['mean_precision'],
            'Mean_Recall': prf['mean_recall'],
            'Mean_F1': prf['mean_f1'],
            'Class_IoU': self.get_class_iou(),
        }


def evaluate_model(
    model_path: str,
    val_data_path: str,
    output_json: str = "evaluation_results.json",
    num_classes: int = 3,
):
    """
    评估模型性能
    
    Args:
        model_path: 模型权重路径
        val_data_path: 验证集路径
        output_json: 输出JSON文件路径
        num_classes: 类别数量
    """
    
    print("=" * 60)
    print("模型评估")
    print("=" * 60)
    
    # 加载模型
    print(f"\n加载模型: {model_path}")
    model = YOLO(model_path)
    
    # 初始化评估器
    metrics = SegmentationMetrics(num_classes=num_classes)
    
    # 获取验证集图像
    val_path = Path(val_data_path)
    if val_path.is_dir():
        image_files = list(val_path.glob("*.jpg")) + \
                     list(val_path.glob("*.jpeg")) + \
                     list(val_path.glob("*.png"))
    else:
        raise ValueError(f"验证集路径不存在: {val_data_path}")
    
    print(f"找到 {len(image_files)} 张验证图像")
    
    # 逐个评估
    boundary_f1_scores = []
    
    for img_path in tqdm(image_files, desc="评估进度"):
        # 读取图像
        image = cv2.imread(str(img_path))
        
        # 预测
        results = model.predict(str(img_path), verbose=False)
        
        if len(results) > 0:
            result = results[0]
            
            # 获取预测mask
            if result.masks is not None:
                pred_mask = result.masks.data[0].cpu().numpy()
                pred_mask = cv2.resize(pred_mask, (image.shape[1], image.shape[0]))
                pred_mask = (pred_mask > 0.5).astype(np.uint8)
                
                # 这里需要真实标签，如果没有则跳过
                # gt_mask = load_ground_truth(img_path)  # 需要实现
                # metrics.update(pred_mask, gt_mask)
                
    # 获取所有指标
    all_metrics = metrics.get_all_metrics()
    
    # 打印结果
    print("\n" + "=" * 60)
    print("评估结果:")
    print("=" * 60)
    print(f"mIoU (Mean IoU):          {all_metrics['mIoU']:.4f}")
    print(f"Dice Coefficient:         {all_metrics['Dice_Coefficient']:.4f}")
    print(f"Pixel Accuracy:           {all_metrics['Pixel_Accuracy']:.4f}")
    print(f"Mean Precision:           {all_metrics['Mean_Precision']:.4f}")
    print(f"Mean Recall:              {all_metrics['Mean_Recall']:.4f}")
    print(f"Mean F1-Score:            {all_metrics['Mean_F1']:.4f}")
    print("\n每类IoU:")
    for class_name, iou in all_metrics['Class_IoU'].items():
        print(f"  {class_name}: {iou:.4f}")
    
    # 保存结果
    with open(output_json, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    print(f"\n结果已保存: {output_json}")
    print("=" * 60)
    
    return all_metrics


def main():
    """命令行接口"""
    parser = argparse.ArgumentParser(description="评估YOLO分割模型")
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="模型权重路径"
    )
    
    parser.add_argument(
        "--val-data",
        type=str,
        required=True,
        help="验证集路径"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results.json",
        help="输出JSON文件"
    )
    
    parser.add_argument(
        "--num-classes",
        type=int,
        default=3,
        help="类别数量"
    )
    
    args = parser.parse_args()
    
    evaluate_model(
        model_path=args.model,
        val_data_path=args.val_data,
        output_json=args.output,
        num_classes=args.num_classes,
    )


if __name__ == "__main__":
    main()
