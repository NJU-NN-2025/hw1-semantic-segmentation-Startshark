"""
数据预处理和SAM伪标签生成脚本
LLM 辅助: 本文件由 GitHub Copilot 辅助生成
"""

import os
import cv2
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import json
from typing import List, Dict
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import shutil


class AntarcticDataPreprocessor:
    """南极动物数据集预处理器"""
    
    def __init__(self, data_dir: str = "dataset", output_dir: str = "data/processed"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 动物类别定义（根据实际数据集调整）
        self.classes = {
            0: 'background',
            1: 'penguin',      # 企鹅
            2: 'seal',         # 海狮/海豹
            3: 'bird',         # 其他鸟类
        }
        
    def organize_dataset(self, train_ratio: float = 0.8):
        """
        整理数据集结构
        dataset/
        ├── images/
        └── labels/  (将由SAM生成)
        """
        print("正在整理数据集结构...")
        
        # 创建目录结构
        for split in ['train', 'val']:
            (self.output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        # 收集所有图像
        image_files = list(self.data_dir.glob('**/*.jpg')) + \
                     list(self.data_dir.glob('**/*.jpeg')) + \
                     list(self.data_dir.glob('**/*.png'))
        
        print(f"找到 {len(image_files)} 张图像")
        
        # 划分训练集和验证集
        np.random.seed(42)
        np.random.shuffle(image_files)
        split_idx = int(len(image_files) * train_ratio)
        
        train_files = image_files[:split_idx]
        val_files = image_files[split_idx:]
        
        # 复制文件
        self._copy_files(train_files, 'train')
        self._copy_files(val_files, 'val')
        
        print(f"训练集: {len(train_files)} 张")
        print(f"验证集: {len(val_files)} 张")
        
    def _copy_files(self, files: List[Path], split: str):
        """复制图像文件到目标目录"""
        for i, file in enumerate(tqdm(files, desc=f"复制{split}集")):
            dest = self.output_dir / split / 'images' / f"{split}_{i:04d}.jpg"
            shutil.copy(file, dest)
    
    def generate_sam_labels(self, 
                           sam_checkpoint: str = "sam_vit_b_01ec64.pth",
                           model_type: str = "vit_b"):
        """
        使用SAM生成伪标签
        适用于 RTX 4060 8GB 显存
        """
        print("初始化 SAM 模型...")
        
        # 检查GPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {device}")
        
        # 加载SAM模型 (使用轻量级 vit_b 适配显存)
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        
        # 配置自动mask生成器 - 降低参数适配8GB显存
        mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=16,  # 降低采样密度
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            crop_n_layers=0,  # 不使用crop减少显存占用
            crop_n_points_downscale_factor=1,
            min_mask_region_area=100,
        )
        
        # 处理训练集和验证集
        for split in ['train', 'val']:
            self._generate_labels_for_split(mask_generator, split)
    
    def _generate_labels_for_split(self, mask_generator, split: str):
        """为指定分割生成标签"""
        image_dir = self.output_dir / split / 'images'
        label_dir = self.output_dir / split / 'labels'
        
        image_files = sorted(image_dir.glob('*.jpg'))
        print(f"\n处理 {split} 集 ({len(image_files)} 张图像)...")
        
        for img_path in tqdm(image_files, desc=f"生成{split}集标签"):
            # 读取图像
            image = cv2.imread(str(img_path))
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 生成masks
            try:
                masks = mask_generator.generate(image_rgb)
                
                # 转换为YOLO segmentation格式
                self._save_yolo_labels(masks, img_path, label_dir, image.shape[:2])
                
            except Exception as e:
                print(f"\n处理 {img_path.name} 时出错: {e}")
                continue
    
    def _save_yolo_labels(self, masks: List[Dict], img_path: Path, 
                         label_dir: Path, img_shape: tuple):
        """
        将SAM masks保存为YOLO segmentation格式
        格式: class_id x1 y1 x2 y2 ... xn yn (归一化坐标)
        """
        h, w = img_shape
        label_path = label_dir / f"{img_path.stem}.txt"
        
        with open(label_path, 'w') as f:
            for mask_data in masks:
                # 简单分类策略: 根据mask大小和位置分配类别
                class_id = self._assign_class(mask_data, (h, w))
                
                # 获取mask轮廓
                segmentation = mask_data['segmentation']
                contours, _ = cv2.findContours(
                    segmentation.astype(np.uint8), 
                    cv2.RETR_EXTERNAL, 
                    cv2.CHAIN_APPROX_SIMPLE
                )
                
                if len(contours) == 0:
                    continue
                
                # 使用最大轮廓
                contour = max(contours, key=cv2.contourArea)
                
                # 归一化坐标
                points = contour.reshape(-1, 2)
                points_norm = points.copy().astype(float)
                points_norm[:, 0] /= w
                points_norm[:, 1] /= h
                
                # 写入YOLO格式
                line = f"{class_id}"
                for point in points_norm:
                    line += f" {point[0]:.6f} {point[1]:.6f}"
                f.write(line + '\n')
    
    def _assign_class(self, mask_data: Dict, img_shape: tuple) -> int:
        """
        简单的类别分配策略（可以根据实际情况改进）
        这里使用mask的面积和位置进行分类
        """
        bbox = mask_data['bbox']  # [x, y, w, h]
        area = mask_data['area']
        h, w = img_shape
        
        # 基于面积的简单分类
        relative_area = area / (h * w)
        
        if relative_area > 0.1:
            return 1  # 大动物 - 企鹅
        elif relative_area > 0.02:
            return 2  # 中等动物 - 海狮
        else:
            return 3  # 小动物 - 鸟类
    
    def create_yaml_config(self):
        """创建YOLO训练配置文件"""
        yaml_content = f"""# 南极动物语义分割数据集配置
# LLM 辅助生成

path: {self.output_dir.absolute()}
train: train/images
val: val/images

# 类别数量
nc: {len(self.classes) - 1}  # 不包含background

# 类别名称
names:
  0: penguin
  1: seal
  2: bird
"""
        
        yaml_path = self.output_dir / 'antarctic.yaml'
        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write(yaml_content)
        
        print(f"\n配置文件已保存: {yaml_path}")


def main():
    """主函数"""
    print("=" * 60)
    print("南极动物语义分割 - 数据预处理")
    print("=" * 60)
    
    # 初始化预处理器
    preprocessor = AntarcticDataPreprocessor(
        data_dir="dataset",
        output_dir="data/processed"
    )
    
    # 步骤1: 整理数据集
    preprocessor.organize_dataset(train_ratio=0.8)
    
    # 步骤2: 生成YAML配置
    preprocessor.create_yaml_config()
    
    # 步骤3: 使用SAM生成伪标签
    print("\n" + "=" * 60)
    print("准备生成SAM伪标签...")
    print("请确保已下载SAM模型权重: sam_vit_b_01ec64.pth")
    print("下载链接: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth")
    print("=" * 60)
    
    use_sam = input("\n是否使用SAM生成伪标签? (y/n, 需要GPU): ").lower()
    
    if use_sam == 'y':
        sam_checkpoint = input("SAM模型路径 (默认: sam_vit_b_01ec64.pth): ").strip()
        if not sam_checkpoint:
            sam_checkpoint = "sam_vit_b_01ec64.pth"
        
        if not os.path.exists(sam_checkpoint):
            print(f"错误: 找不到模型文件 {sam_checkpoint}")
            print("请先下载SAM模型")
        else:
            preprocessor.generate_sam_labels(sam_checkpoint=sam_checkpoint)
    
    print("\n" + "=" * 60)
    print("数据预处理完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
