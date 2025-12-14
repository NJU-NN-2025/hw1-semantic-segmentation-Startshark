import cv2
import os
import numpy as np
from pathlib import Path
import math

def stitch_images(image_dir, output_path, grid_size=(4, 4)):
    """
    Stitch images from a directory into a grid.
    """
    image_dir = Path(image_dir)
    output_path = Path(output_path)
    
    # Get all jpg images
    images = sorted(list(image_dir.glob("*.jpg")))
    
    if not images:
        print(f"No images found in {image_dir}")
        return

    print(f"Found {len(images)} images.")
    
    # Take only up to grid_size[0] * grid_size[1] images
    max_images = grid_size[0] * grid_size[1]
    images = images[:max_images]
    
    # Read images
    img_list = []
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is not None:
            img_list.append(img)
        else:
            print(f"Failed to read {img_path}")

    if not img_list:
        print("No valid images to stitch.")
        return

    # Assume all images are the same size as the first one
    h, w, c = img_list[0].shape
    
    # Create canvas
    grid_h = grid_size[0] * h
    grid_w = grid_size[1] * w
    canvas = np.zeros((grid_h, grid_w, c), dtype=np.uint8)
    
    for idx, img in enumerate(img_list):
        # Resize if necessary (though we assume same size for now)
        if img.shape != (h, w, c):
            img = cv2.resize(img, (w, h))
            
        row = idx // grid_size[1]
        col = idx % grid_size[1]
        
        y_start = row * h
        y_end = y_start + h
        x_start = col * w
        x_end = x_start + w
        
        canvas[y_start:y_end, x_start:x_end] = img
        
    # Save result
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)
    print(f"Stitched image saved to {output_path}")

if __name__ == "__main__":
    input_dir = r"c:\Users\Jason Chen\Desktop\Code\hw1-semantic-segmentation-Startshark\runs\segment\antarctic_yolo_v9\test_results\predictions2"
    output_file = r"c:\Users\Jason Chen\Desktop\Code\hw1-semantic-segmentation-Startshark\runs\segment\antarctic_yolo_v9\test_results\stitched_predictions.jpg"
    
    stitch_images(input_dir, output_file)
