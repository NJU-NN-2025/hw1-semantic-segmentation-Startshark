import json
import os
import shutil
import yaml
import argparse
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np

def convert_coco_to_yolo_segmentation(json_path, images_dir, output_images_dir, output_labels_dir):
    """
    Convert COCO JSON annotations to YOLO segmentation format.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    images = {img['id']: img for img in data['images']}
    categories = {cat['id']: cat['name'] for cat in data['categories']}
    
    # Create category mapping (COCO ID -> YOLO ID 0-indexed)
    # We sort categories by ID to ensure consistent mapping
    sorted_cats = sorted(data['categories'], key=lambda x: x['id'])
    cat_id_map = {cat['id']: i for i, cat in enumerate(sorted_cats)}
    yolo_classes = {i: cat['name'] for i, cat in enumerate(sorted_cats)}

    # Group annotations by image_id
    img_anns = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_anns:
            img_anns[img_id] = []
        img_anns[img_id].append(ann)

    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    for img_id, img_info in tqdm(images.items(), desc=f"Converting {json_path.parent.name}"):
        file_name = img_info['file_name']
        src_img_path = images_dir / file_name
        
        if not src_img_path.exists():
            print(f"Warning: Image {src_img_path} not found. Skipping.")
            continue

        # Copy image
        dst_img_path = output_images_dir / file_name
        shutil.copy2(src_img_path, dst_img_path)

        # Create label file
        label_file = output_labels_dir / f"{Path(file_name).stem}.txt"
        
        img_w = img_info['width']
        img_h = img_info['height']

        with open(label_file, 'w') as f:
            if img_id in img_anns:
                for ann in img_anns[img_id]:
                    cat_id = ann['category_id']
                    if cat_id not in cat_id_map:
                        continue
                    
                    yolo_cat_id = cat_id_map[cat_id]
                    
                    # Handle segmentation
                    if 'segmentation' in ann:
                        segmentations = ann['segmentation']
                        if not segmentations:
                            continue
                            
                        # Check if it's RLE (dict) or Polygon (list)
                        if isinstance(segmentations, dict):
                            print(f"Warning: RLE format not supported for {file_name}. Skipping annotation.")
                            continue
                            
                        # Check if it's a single polygon (list of floats) or list of polygons (list of lists)
                        # Standard COCO is list of lists: [[x1, y1, ...], [x1, y1, ...]]
                        # But sometimes it might be just [x1, y1, ...] if not standard? 
                        # Actually standard is always list of lists.
                        # If iterating over it gives numbers, then it's a single list.
                        
                        if isinstance(segmentations[0], (int, float)):
                            # It's a single list of coordinates
                            polygons = [segmentations]
                        else:
                            # It's a list of lists
                            polygons = segmentations

                        for seg in polygons:
                            if len(seg) < 4: # Need at least 2 points
                                continue
                                
                            # COCO segmentation is [x1, y1, x2, y2, ...]
                            # YOLO expects normalized [x1, y1, x2, y2, ...]
                            try:
                                points = np.array(seg).reshape(-1, 2)
                            except ValueError:
                                print(f"Warning: Malformed segmentation in {file_name}. Skipping.")
                                continue
                            
                            # Normalize
                            points[:, 0] /= img_w
                            points[:, 1] /= img_h
                            
                            # Clip to [0, 1]
                            points = np.clip(points, 0, 1)
                            
                            # Write to file
                            line = f"{yolo_cat_id}"
                            for pt in points:
                                line += f" {pt[0]:.6f} {pt[1]:.6f}"
                            f.write(line + "\n")
    
    return yolo_classes

def main():
    parser = argparse.ArgumentParser(description="Convert Roboflow COCO format to YOLOv8 Segmentation format")
    parser.add_argument("--source", type=str, required=True, help="Path to Roboflow dataset root (e.g., 'Polar Animal.v14')")
    parser.add_argument("--output", type=str, default="data/processed", help="Output directory")
    args = parser.parse_args()

    source_dir = Path(args.source)
    output_dir = Path(args.output)

    # Define splits mapping
    splits = {
        'train': 'train',
        'valid': 'val',
        'test': 'test'
    }

    all_classes = {}

    for src_split, dst_split in splits.items():
        split_dir = source_dir / src_split
        if not split_dir.exists():
            print(f"Split {src_split} not found in {source_dir}. Skipping.")
            continue

        json_file = split_dir / "_annotations.coco.json"
        if not json_file.exists():
            print(f"Annotation file not found in {split_dir}. Skipping.")
            continue

        print(f"Processing {src_split} set...")
        classes = convert_coco_to_yolo_segmentation(
            json_file,
            split_dir, # Images are in the same dir as json in Roboflow export usually, or check structure
            output_dir / dst_split / 'images',
            output_dir / dst_split / 'labels'
        )
        if classes:
            all_classes = classes

    # Generate dataset.yaml
    yaml_content = {
        'path': str(output_dir.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'names': all_classes
    }

    yaml_path = output_dir / 'antarctic.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False)
    
    print(f"\nConversion complete. YAML config saved to {yaml_path}")
    print(f"Classes detected: {all_classes}")

if __name__ == "__main__":
    main()
