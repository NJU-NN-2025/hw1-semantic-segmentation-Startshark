import os
import cv2
import numpy as np
from pathlib import Path

def stitch_appendix_images():
    input_dir = Path("test")
    output_dir = Path("runs/segment/antarctic_yolo_v9/appendix_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get original images (ending in .jpg)
    original_images = sorted(list(input_dir.glob("*.jpg")))
    
    if not original_images:
        print("No jpg images found in test/")
        return

    rows = []
    target_height = 300 # Reasonable height for each row to fit on page

    for img_path in original_images:
        base_name = img_path.stem # e.g., 5N0A5174_1
        
        # Define paths for the triplet
        orig_path = img_path
        mask_path = input_dir / f"{base_name}_mask.png"
        result_path = input_dir / f"{base_name}_result.png"
        
        # Read images
        img_orig = cv2.imread(str(orig_path))
        img_mask = cv2.imread(str(mask_path))
        img_result = cv2.imread(str(result_path))
        
        if img_orig is None:
            print(f"Warning: Could not read {orig_path}")
            continue
            
        row_images = [img_orig]
        
        # Check if mask exists
        if img_mask is not None:
            row_images.append(img_mask)
        else:
            print(f"Warning: Missing mask for {base_name}, using black placeholder")
            row_images.append(np.zeros_like(img_orig))

        # Check if result exists
        if img_result is not None:
            row_images.append(img_result)
        else:
            print(f"Warning: Missing result for {base_name}, using black placeholder")
            row_images.append(np.zeros_like(img_orig))

        # Resize all to target height
        resized_row = []
        for img in row_images:
            h, w = img.shape[:2]
            scale = target_height / h
            new_w = int(w * scale)
            resized = cv2.resize(img, (new_w, target_height))
            resized_row.append(resized)
            
        # Stitch horizontally with some spacing
        spacer = np.ones((target_height, 10, 3), dtype=np.uint8) * 255
        row_stitched = resized_row[0]
        for i in range(1, len(resized_row)):
            row_stitched = np.hstack([row_stitched, spacer, resized_row[i]])
            
        rows.append(row_stitched)

    if rows:
        # Find max width to pad rows if necessary
        max_width = max(row.shape[1] for row in rows)
        
        final_rows = []
        vertical_spacer = np.ones((10, max_width, 3), dtype=np.uint8) * 255
        
        for i, row in enumerate(rows):
            h, w = row.shape[:2]
            if w < max_width:
                pad = np.ones((h, max_width - w, 3), dtype=np.uint8) * 255
                row = np.hstack([row, pad])
            
            if i > 0:
                final_rows.append(vertical_spacer)
            final_rows.append(row)

        # Stitch vertically
        final_image = np.vstack(final_rows)
        
        output_path = output_dir / "stitched_appendix.jpg"
        cv2.imwrite(str(output_path), final_image)
        print(f"Saved {output_path}")

if __name__ == "__main__":
    stitch_appendix_images()
