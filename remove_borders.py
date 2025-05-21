#!/usr/bin/env python3
"""
Border Trimmer: A tool to automatically remove white/blank borders from comic panel images.
Aggressively crops all whitespace up to the black panel outlines.

Dependencies:
  - opencv-python (for image processing)
  - numpy (for array operations)
  - tqdm (for progress bars)

Install:
  pip install opencv-python numpy tqdm
"""

import os
import cv2
import numpy as np
import glob
from pathlib import Path
import time
from tqdm import tqdm
import argparse

def detect_borders(img, threshold=220, tolerance=5):
    """
    Detect white borders around an image, cropping aggressively right up to the black panel outline.
    Returns (top, right, bottom, left) with the number of border pixels on each side.
    """
    if len(img.shape) == 3:
        # Convert to grayscale if it's a color image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    height, width = gray.shape
    
    # Check if the image has an alpha channel (RGBA)
    has_alpha = False
    if len(img.shape) == 3 and img.shape[2] == 4:
        has_alpha = True
        # Get alpha channel
        alpha = img[:, :, 3]
    
    # Initialize border values
    top = 0
    bottom = 0
    left = 0
    right = 0
    
    # Create multiple binary images at different thresholds for better detection
    _, binary_strict = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    _, binary_relaxed = cv2.threshold(gray, threshold - 20, 255, cv2.THRESH_BINARY)
    
    # Create edge detection image to find panel outlines
    edges = cv2.Canny(gray, 50, 150)
    
    # Dilate edges to make them more prominent
    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    
    # Create a gradient magnitude image to detect borders
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = cv2.magnitude(sobelx, sobely)
    gradient_mag = cv2.normalize(gradient_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Find top border
    for y in range(height):
        # Stop at any strong edge
        if np.any(dilated_edges[y, :] > 0):
            break
            
        # Stop at any strong gradient (indicates transition to panel)
        if np.max(gradient_mag[y, :]) > 50:
            break
            
        # Check whiteness using strict binary
        row = binary_strict[y, :]
        white_pixels = np.sum(row > 0) / len(row)
        
        # If row is not mostly white, check using relaxed threshold
        if white_pixels < 0.95:
            row_relaxed = binary_relaxed[y, :]
            white_pixels_relaxed = np.sum(row_relaxed > 0) / len(row_relaxed)
            if white_pixels_relaxed < 0.98:
                break
        
        top = y + 1
    
    # Find bottom border
    for y in range(height - 1, -1, -1):
        # Stop at any strong edge
        if np.any(dilated_edges[y, :] > 0):
            break
            
        # Stop at any strong gradient
        if np.max(gradient_mag[y, :]) > 50:
            break
        
        # Check whiteness using strict binary
        row = binary_strict[y, :]
        white_pixels = np.sum(row > 0) / len(row)
        
        # If row is not mostly white, check using relaxed threshold
        if white_pixels < 0.95:
            row_relaxed = binary_relaxed[y, :]
            white_pixels_relaxed = np.sum(row_relaxed > 0) / len(row_relaxed)
            if white_pixels_relaxed < 0.98:
                break
        
        bottom = height - y
    
    # Find left border - with handling of adjacent panel slivers
    for x in range(width):
        col = gray[:, x]
        
        # Check if this column contains a thin vertical line (potential panel edge)
        if np.any(dilated_edges[:, x] > 0):
            # Look ahead to see if there's whitespace after this edge (sliver detection)
            look_ahead = min(30, width - x - 1)  # Increased from 15 to 30 pixels ahead
            if look_ahead > 3:  # Only if we have enough pixels to analyze
                # Check if we have a pattern of: panel edge followed by mostly white
                after_edge = binary_strict[:, x+1:x+look_ahead]
                if after_edge.size > 0:
                    white_ratio_after = np.sum(after_edge > 0) / after_edge.size
                    # If there's whitespace after the edge, this might be a sliver or larger intrusion
                    if white_ratio_after > 0.7:  # Reduced from 0.8 to handle less white areas
                        # Continue searching (skip this edge)
                        continue
                    
                    # Additional check: is there another black line after this whitespace?
                    # This would indicate another panel's border in the distance
                    extended_look = min(60, width - x - look_ahead - 1)
                    if extended_look > 10:
                        far_section = gray[:, x+look_ahead:x+look_ahead+extended_look]
                        if far_section.size > 0:
                            # Check if there's another dark region (another panel border)
                            dark_pixels = np.sum(far_section < 100) / far_section.size
                            if dark_pixels > 0.05:  # If there are some dark pixels (potential border)
                                # This is likely another panel border in the distance, skip current edge
                                continue
            
            # Not a sliver or we've reached a substantial edge, stop here
            break
            
        # Stop at any strong gradient
        if np.max(gradient_mag[:, x]) > 50:
            # Check if this is a thin line by looking for gradient followed by whitespace
            look_ahead = min(20, width - x - 1)  # Increased from 10 to 20
            if look_ahead > 3:
                after_gradient = binary_relaxed[:, x+1:x+look_ahead]
                if after_gradient.size > 0:
                    white_ratio_after = np.sum(after_gradient > 0) / after_gradient.size
                    if white_ratio_after > 0.7:  # Reduced from 0.8 to be more aggressive
                        # This is likely a thin line followed by whitespace (sliver)
                        continue
            break
        
        # Check whiteness using strict binary
        col = binary_strict[:, x]
        white_pixels = np.sum(col > 0) / len(col)
        
        # If column is not mostly white, check using relaxed threshold
        if white_pixels < 0.95:
            col_relaxed = binary_relaxed[:, x]
            white_pixels_relaxed = np.sum(col_relaxed > 0) / len(col_relaxed)
            if white_pixels_relaxed < 0.98:
                break
        
        left = x + 1
    
    # Find right border - with handling of adjacent panel slivers
    for x in range(width - 1, -1, -1):
        col = gray[:, x]
        
        # Check if this column contains a thin vertical line (potential panel edge)
        if np.any(dilated_edges[:, x] > 0):
            # Look ahead (to the left) to see if there's whitespace after this edge
            look_ahead = min(30, x)  # Increased from 15 to 30 pixels ahead
            if look_ahead > 3:  # Only if we have enough pixels to analyze
                # Check if we have a pattern of: panel edge followed by mostly white
                after_edge = binary_strict[:, x-look_ahead:x]
                if after_edge.size > 0:
                    white_ratio_after = np.sum(after_edge > 0) / after_edge.size
                    # If there's whitespace after the edge, this might be a sliver or larger intrusion
                    if white_ratio_after > 0.7:  # Reduced from 0.8 to handle less white areas
                        # Continue searching (skip this edge)
                        continue
                        
                    # Additional check: is there another black line after this whitespace?
                    # This would indicate another panel's border in the distance
                    extended_look = min(60, x - look_ahead)
                    if extended_look > 10:
                        far_section = gray[:, x-look_ahead-extended_look:x-look_ahead]
                        if far_section.size > 0:
                            # Check if there's another dark region (another panel border)
                            dark_pixels = np.sum(far_section < 100) / far_section.size
                            if dark_pixels > 0.05:  # If there are some dark pixels (potential border)
                                # This is likely another panel border in the distance, skip current edge
                                continue
            
            # Not a sliver or we've reached a substantial edge, stop here
            break
            
        # Stop at any strong gradient
        if np.max(gradient_mag[:, x]) > 50:
            # Check if this is a thin line by looking for gradient followed by whitespace
            look_ahead = min(20, x)  # Increased from 10 to 20
            if look_ahead > 3:
                after_gradient = binary_relaxed[:, x-look_ahead:x]
                if after_gradient.size > 0:
                    white_ratio_after = np.sum(after_gradient > 0) / after_gradient.size
                    if white_ratio_after > 0.7:  # Reduced from 0.8 to be more aggressive
                        # This is likely a thin line followed by whitespace (sliver)
                        continue
            break
        
        # Check whiteness using strict binary
        col = binary_strict[:, x]
        white_pixels = np.sum(col > 0) / len(col)
        
        # If column is not mostly white, check using relaxed threshold
        if white_pixels < 0.95:
            col_relaxed = binary_relaxed[:, x]
            white_pixels_relaxed = np.sum(col_relaxed > 0) / len(col_relaxed)
            if white_pixels_relaxed < 0.98:
                break
        
        right = width - x
    
    # For panels where we didn't detect any border, try a different approach
    # This happens with some slightly off-white borders
    if left == 0 and right == 0 and top == 0 and bottom == 0:
        # Use histogram-based approach
        row_means = np.mean(gray, axis=1)  # Average brightness of each row
        col_means = np.mean(gray, axis=0)  # Average brightness of each column
        
        # Find drops in brightness (indicating edge of whitespace)
        for y in range(1, height):
            if row_means[y] < row_means[y-1] - 5:  # Brightness drop
                top = y
                break
        
        for y in range(height - 2, -1, -1):
            if row_means[y] < row_means[y+1] - 5:  # Brightness drop
                bottom = height - y - 1
                break
        
        for x in range(1, width):
            if col_means[x] < col_means[x-1] - 5:  # Brightness drop
                left = x
                break
        
        for x in range(width - 2, -1, -1):
            if col_means[x] < col_means[x+1] - 5:  # Brightness drop
                right = width - x - 1
                break
    
    # Safety check - make sure we're not cropping too much
    # Ensure at least 70% of the image remains
    if (height - top - bottom) < 0.7 * height or (width - left - right) < 0.7 * width:
        # Reset to more conservative values if we're cropping too aggressively
        if (height - top - bottom) < 0.7 * height:
            excess = (0.7 * height) - (height - top - bottom)
            # Distribute the reduction between top and bottom
            top_reduce = int(excess / 2)
            bottom_reduce = int(excess / 2)
            top = max(0, top - top_reduce)
            bottom = max(0, bottom - bottom_reduce)
            
        if (width - left - right) < 0.7 * width:
            excess = (0.7 * width) - (width - left - right)
            # Distribute the reduction between left and right
            left_reduce = int(excess / 2)
            right_reduce = int(excess / 2)
            left = max(0, left - left_reduce)
            right = max(0, right - right_reduce)
    
    return (top, right, bottom, left)

def crop_image(img, borders):
    """
    Crop an image based on detected borders.
    """
    top, right, bottom, left = borders
    height, width = img.shape[:2]
    
    # Ensure we don't crop the entire image
    if top + bottom >= height or left + right >= width:
        return img
    
    # Crop the image
    cropped = img[top:height-bottom, left:width-right]
    return cropped

def process_image(input_path, output_path, threshold=220, tolerance=5, debug=False):
    """
    Process a single image to remove borders.
    """
    try:
        # Read image with alpha channel if present
        img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Error: Could not read image {input_path}")
            return False
        
        # Detect borders
        borders = detect_borders(img, threshold, tolerance)
        top, right, bottom, left = borders
        
        # If in debug mode, save an image showing the detected borders
        if debug:
            debug_img = img.copy()
            height, width = img.shape[:2]
            
            # Draw top border
            if top > 0:
                cv2.rectangle(debug_img, (0, 0), (width, top), (0, 0, 255), 2)
            
            # Draw bottom border
            if bottom > 0:
                cv2.rectangle(debug_img, (0, height-bottom), (width, height), (0, 0, 255), 2)
            
            # Draw left border
            if left > 0:
                cv2.rectangle(debug_img, (0, 0), (left, height), (0, 255, 0), 2)
            
            # Draw right border
            if right > 0:
                cv2.rectangle(debug_img, (width-right, 0), (width, height), (0, 255, 0), 2)
            
            # Save debug image
            debug_path = output_path.replace('.png', '_border_debug.png')
            cv2.imwrite(debug_path, debug_img)
            
            # Print detected border sizes
            print(f"{input_path}: borders = top:{top}, right:{right}, bottom:{bottom}, left:{left}")
        
        # Crop image
        cropped = crop_image(img, borders)
        
        # Save the result
        cv2.imwrite(output_path, cropped)
        
        # Return true if any cropping was done
        return any(b > 0 for b in borders)
        
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False

def process_directory(input_dir, output_dir, threshold=220, tolerance=5, limit=None, debug=False):
    """
    Process all images in a directory.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all PNG files in the input directory
    image_files = glob.glob(os.path.join(input_dir, "**", "*.png"), recursive=True)
    
    # Limit the number of files if specified
    if limit is not None and limit > 0:
        image_files = image_files[:limit]
    
    print(f"Found {len(image_files)} images to process")
    
    processed_count = 0
    cropped_count = 0
    
    # Process each image
    for img_path in tqdm(image_files, desc="Processing images"):
        # Create the same directory structure in the output directory
        rel_path = os.path.relpath(img_path, input_dir)
        out_path = os.path.join(output_dir, rel_path)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        
        # Process the image
        was_cropped = process_image(img_path, out_path, threshold, tolerance, debug)
        
        processed_count += 1
        if was_cropped:
            cropped_count += 1
    
    print(f"Processed {processed_count} images")
    print(f"Cropped borders from {cropped_count} images")

def main():
    parser = argparse.ArgumentParser(description="Remove white borders from comic panels")
    parser.add_argument("--input", default="panels", help="Input directory containing panel images")
    parser.add_argument("--output", default="panels_trimmed", help="Output directory for processed images")
    parser.add_argument("--threshold", type=int, default=220, help="Brightness threshold for detecting borders (0-255)")
    parser.add_argument("--tolerance", type=int, default=5, help="Percentage of non-white pixels allowed in border (0-100)")
    parser.add_argument("--limit", type=int, default=None, help="Limit processing to this many images (for testing)")
    parser.add_argument("--debug", action="store_true", help="Save debug images showing detected borders")
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    process_directory(args.input, args.output, args.threshold, args.tolerance, args.limit, args.debug)
    
    elapsed_time = time.time() - start_time
    print(f"Total processing time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main() 