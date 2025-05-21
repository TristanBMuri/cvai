#!/usr/bin/env python3
"""
Enhanced Speech Bubble Remover: An improved tool to detect and remove speech bubbles from comic panels.
Handles speech bubbles differently based on their location:
- Top speech bubbles are cropped off completely
- Interior speech bubbles can be made transparent or inpainted

Dependencies:
  - opencv-python (for image processing)
  - numpy (for array operations)
  - easyocr (for OCR)
  - pillow (for image operations)

Install:
  pip install opencv-python numpy easyocr pillow tqdm
"""

import os
import cv2
import numpy as np
import easyocr
from PIL import Image
import glob
from pathlib import Path
import time
from tqdm import tqdm
import argparse

# Initialize EasyOCR reader (only do this once)
reader = easyocr.Reader(['en'], gpu=False)

def detect_text_regions(image_path):
    """
    Use OCR to detect regions of text in an image (speech bubbles).
    Returns bounding boxes of detected text regions.
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return [], None
    
    # Use EasyOCR to detect text regions
    # EasyOCR works better on the original image
    results = reader.readtext(img)
    
    # Extract bounding boxes of detected text
    boxes = []
    for result in results:
        # Each result is ([top-left, top-right, bottom-right, bottom-left], text, confidence)
        bbox = result[0]
        text = result[1]
        confidence = result[2]
        
        # Only process if confidence is high enough and text is not too short
        if confidence > 0.2 and len(text.strip()) > 1:
            # Convert points to x, y, w, h format
            x_min = min(point[0] for point in bbox)
            y_min = min(point[1] for point in bbox)
            x_max = max(point[0] for point in bbox)
            y_max = max(point[1] for point in bbox)
            
            w = x_max - x_min
            h = y_max - y_min
            
            # Skip very small regions that are likely noise
            if w < 10 or h < 10:
                continue
                
            boxes.append((int(x_min), int(y_min), int(w), int(h)))
    
    return boxes, img

def detect_speech_bubbles_with_whitespace(img):
    """
    Detect speech bubbles by looking for large white areas.
    This can help catch bubbles that OCR might miss.
    """
    height, width = img.shape[:2]
    panel_area = height * width
    max_bubble_area = panel_area * 0.7  # Max 70% of panel area
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Threshold the image to find white areas
    _, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
    
    # Find contours of white areas
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by size and shape
    bubble_boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Skip if too small
        if area < 500:
            continue
            
        # Skip if too large (more than 70% of panel)
        if area > max_bubble_area:
            continue
            
        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)
        
        # Skip if bounding box is too large
        if w * h > max_bubble_area:
            continue
            
        # Calculate some metrics for filtering
        rect_area = w * h
        extent = float(area) / rect_area
        
        # Filter by shape (speech bubbles tend to be somewhat round/rectangular)
        if extent > 0.6 and w > 30 and h > 20:
            bubble_boxes.append((x, y, w, h))
    
    return bubble_boxes

def is_box_contained(box1, box2, threshold=0.8):
    """
    Check if box1 is mostly contained within box2.
    Returns True if the overlap area is more than threshold of box1's area.
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Calculate intersection
    x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    overlap_area = x_overlap * y_overlap
    
    # Calculate area of box1
    box1_area = w1 * h1
    
    # Check if overlap is significant
    return overlap_area > box1_area * threshold and box1_area < w2 * h2

def filter_nested_boxes(boxes):
    """
    Filter out boxes that are mostly contained within other boxes.
    """
    if not boxes:
        return []
        
    # Sort boxes by area (largest first)
    sorted_boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    
    filtered_boxes = [sorted_boxes[0]]  # Keep the largest box
    
    # Check each box against the filtered boxes
    for box in sorted_boxes[1:]:
        is_nested = False
        for filtered_box in filtered_boxes:
            if is_box_contained(box, filtered_box):
                is_nested = True
                break
        
        if not is_nested:
            filtered_boxes.append(box)
    
    return filtered_boxes

def merge_overlapping_boxes(boxes, overlap_threshold=0.3):
    """
    Merge boxes that have significant overlap.
    """
    if not boxes:
        return []
    
    merged_boxes = []
    
    # Sort boxes by x-coordinate
    sorted_boxes = sorted(boxes, key=lambda b: b[0])
    
    # Start with the first box
    current_box = sorted_boxes[0]
    
    for box in sorted_boxes[1:]:
        x1, y1, w1, h1 = current_box
        x2, y2, w2, h2 = box
        
        # Calculate intersection
        x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
        y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
        overlap_area = x_overlap * y_overlap
        
        # Calculate union area
        union_area = w1 * h1 + w2 * h2 - overlap_area
        
        # Check if overlap ratio is significant
        overlap_ratio = overlap_area / union_area if union_area > 0 else 0
        
        if overlap_ratio > overlap_threshold:
            # Merge boxes
            x = min(x1, x2)
            y = min(y1, y2)
            w = max(x1 + w1, x2 + w2) - x
            h = max(y1 + h1, y2 + h2) - y
            current_box = (x, y, w, h)
        else:
            # No significant overlap, add current box and move to next
            merged_boxes.append(current_box)
            current_box = box
    
    # Add the last box
    merged_boxes.append(current_box)
    
    return merged_boxes

def expand_text_regions_to_bubble(img, boxes):
    """
    Expand detected text regions to cover the entire speech bubble.
    Uses flood fill to detect white areas around text, stopping at dark borders.
    Aggressively captures all whitespace around text to ensure proper bubble removal.
    """
    height, width = img.shape[:2]
    panel_area = height * width
    max_bubble_area = panel_area * 0.7  # Max 70% of panel area
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Create binary image using adaptive thresholding to better identify bubble borders
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 11, 2)
    
    # Create a binary image with text regions marked
    mask = np.zeros((height, width), dtype=np.uint8)
    
    expanded_boxes = []
    
    # First, filter out nested boxes and merge overlapping ones
    boxes = filter_nested_boxes(boxes)
    boxes = merge_overlapping_boxes(boxes)
    
    # For each text box, expand to cover the surrounding white area (speech bubble)
    for box in boxes:
        x, y, w, h = box
        
        # Skip if box is too large (likely not a speech bubble but the whole panel)
        if w * h > max_bubble_area:
            continue
        
        # Validate box coordinates
        if x < 0 or y < 0 or x + w > width or y + h > height:
            continue
            
        # If box is too small, skip it
        if w < 15 or h < 15:
            continue
        
        # Expand initial box slightly to better capture the speech bubble
        padding = int(min(w, h) * 0.3)  # Increased from 20% to 30% padding
        x_padded = max(0, x - padding)
        y_padded = max(0, y - padding)
        w_padded = min(width - x_padded, w + 2*padding)
        h_padded = min(height - y_padded, h + 2*padding)
        
        # Create a seed point inside the text box
        seed_x = x + w // 2
        seed_y = y + h // 2
        
        # Create a mask for flood fill
        flood_mask = np.zeros((height + 2, width + 2), dtype=np.uint8)
        
        # Flood fill parameters - use larger tolerance to capture more whitespace
        # This helps ensure all the whitespace around text is included
        flood_fill_flags = 4  # 4-connected
        flood_fill_flags |= cv2.FLOODFILL_FIXED_RANGE
        flood_fill_flags |= (255 << 8)  # Value to fill with
        
        # Perform flood fill to detect bubble area - increased tolerance from 10,30 to 15,50
        cv2.floodFill(gray, flood_mask, (seed_x, seed_y), 
                     255, 15, 50, flood_fill_flags)
        
        # Extract the filled region from the mask
        # Offset by 1 because of the padding in the flood mask
        fill_mask = flood_mask[1:-1, 1:-1]
        
        # Find contours in the filled region
        contours, _ = cv2.findContours(fill_mask, cv2.RETR_EXTERNAL, 
                                      cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get the largest contour (should be the speech bubble)
            largest_contour = max(contours, key=cv2.contourArea)
            contour_area = cv2.contourArea(largest_contour)
            
            # Skip if contour is too large (more than 70% of panel)
            if contour_area > max_bubble_area:
                continue
                
            # Get bounding box of the contour
            x_bubble, y_bubble, w_bubble, h_bubble = cv2.boundingRect(largest_contour)
            
            # Skip if bounding box is too large
            if w_bubble * h_bubble > max_bubble_area:
                continue
            
            # Create a more precise mask by filling only the actual contour
            # rather than using the rectangular bounding box
            contour_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.drawContours(contour_mask, [largest_contour], 0, 255, -1)
            
            # Dilate the contour mask slightly to ensure all whitespace is captured
            kernel = np.ones((3, 3), np.uint8)
            contour_mask = cv2.dilate(contour_mask, kernel, iterations=1)
            
            # Only use if significantly larger than original text box
            # but not excessively large (which might indicate it spilled out too far)
            box_area = w * h
            box_area_padded = w_padded * h_padded
            
            if (contour_area > box_area * 1.2 and  # Reduced from 1.3 to be more inclusive
                contour_area < box_area * 8 and
                contour_area < max_bubble_area and
                w_bubble < width * 0.5 and  # Not more than half the image width
                h_bubble < height * 0.5):   # Not more than half the image height
                
                # Check if the contour is compact (not stretched out along a tail)
                # by comparing area to perimeter squared 
                perimeter = cv2.arcLength(largest_contour, True)
                circularity = 4 * np.pi * contour_area / (perimeter * perimeter) if perimeter > 0 else 0
                
                # If not very circular, check aspect ratio - relaxed from 3.0 to 4.0
                aspect_ratio = max(w_bubble, h_bubble) / min(w_bubble, h_bubble) if min(w_bubble, h_bubble) > 0 else 999
                
                if circularity > 0.15 or aspect_ratio < 4.0:  # Relaxed circularity check
                    # Use the contour mask to include precise bubble shape
                    mask = cv2.bitwise_or(mask, contour_mask)
                    expanded_boxes.append((x_bubble, y_bubble, w_bubble, h_bubble))
                else:
                    # If it's a very elongated contour (possible tail problem),
                    # just use a slightly expanded version of the original text box
                    x_safe = max(0, x - padding)  # Increased padding
                    y_safe = max(0, y - padding)
                    w_safe = min(width - x_safe, w + 2*padding)  # Doubled padding
                    h_safe = min(height - y_safe, h + 2*padding)
                    
                    # Skip if expanded box is too large
                    if w_safe * h_safe > max_bubble_area:
                        continue
                    
                    expanded_boxes.append((x_safe, y_safe, w_safe, h_safe))
                    cv2.rectangle(mask, (x_safe, y_safe), 
                                 (x_safe + w_safe, y_safe + h_safe), 255, -1)
            else:
                # Use the original text box with increased padding
                x_safe = max(0, x - 10)  # Increased from 5 to 10
                y_safe = max(0, y - 10)
                w_safe = min(width - x_safe, w + 20)  # Increased from 10 to 20
                h_safe = min(height - y_safe, h + 20)
                
                # Skip if box with padding is too large
                if w_safe * h_safe > max_bubble_area:
                    continue
                
                expanded_boxes.append((x_safe, y_safe, w_safe, h_safe))
                cv2.rectangle(mask, (x_safe, y_safe), 
                             (x_safe + w_safe, y_safe + h_safe), 255, -1)
    
    # Filter expanded boxes to remove any nested boxes that might have been created
    expanded_boxes = filter_nested_boxes(expanded_boxes)
    
    return expanded_boxes, mask

def classify_speech_bubbles(img, boxes):
    """
    Classify speech bubbles into "top" bubbles (that should be cropped off)
    and "interior" bubbles (that should be made transparent or inpainted).
    
    For top bubbles:
    - Must be in the top portion of the panel
    - Must be wide enough (at least 60% of panel width) to warrant cropping
    - Otherwise, handle as interior bubbles
    
    Returns two lists of boxes: top_bubbles and interior_bubbles.
    """
    height, width = img.shape[:2]
    panel_area = height * width
    max_bubble_area = panel_area * 0.7  # Max 70% of panel area
    
    top_bubbles = []
    interior_bubbles = []
    
    # Define what constitutes a "top" bubble (near the top edge of the panel)
    top_threshold = height * 0.25  # Consider top 25% as the "top" region
    min_width_ratio = 0.6  # Minimum width ratio (to panel width) to consider cropping
    
    for box in boxes:
        x, y, w, h = box
        
        # Skip if box is too large (more than 70% of panel)
        if w * h > max_bubble_area:
            continue
        
        # Calculate width ratio compared to panel
        width_ratio = w / width
        
        # If the box is in the top region
        if y < top_threshold and h > 20:
            # Calculate how much of the box is in the top region
            overlap_height = min(top_threshold, y + h) - y
            overlap_ratio = overlap_height / h
            
            # Only crop if it's both substantially in the top region AND wide enough
            if overlap_ratio > 0.5 and width_ratio >= min_width_ratio:
                top_bubbles.append(box)
            else:
                # Not wide enough to crop, handle as interior bubble
                interior_bubbles.append(box)
        else:
            interior_bubbles.append(box)
    
    return top_bubbles, interior_bubbles

def make_bubbles_transparent(img, mask):
    """
    Make speech bubbles transparent by setting alpha channel.
    """
    # Convert to RGBA
    bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    
    # Set alpha channel to 0 (transparent) where mask is > 0
    bgra[:, :, 3] = 255  # Make everything opaque first
    bgra[:, :, 3] = np.where(mask > 0, 0, 255)  # Make mask areas transparent
    
    return bgra

def inpaint_bubbles(img, mask):
    """
    Inpaint speech bubbles using OpenCV's inpainting.
    """
    # Convert mask to proper format for inpainting
    inpaint_mask = mask.astype(np.uint8)
    
    # Inpaint the masked regions
    inpainted = cv2.inpaint(img, inpaint_mask, 3, cv2.INPAINT_TELEA)
    
    return inpainted

def crop_top_bubbles(img, top_boxes):
    """
    Crop off the top part of the image containing speech bubbles.
    """
    if not top_boxes:
        return img
        
    height, width = img.shape[:2]
    
    # Find the lowest point of all top bubbles
    max_y = 0
    for x, y, w, h in top_boxes:
        max_y = max(max_y, y + h)
    
    # Add some padding to make sure we crop everything
    max_y = min(height, max_y + 10)
    
    # Crop the image
    cropped = img[max_y:, :]
    
    return cropped

def fill_enclosed_regions(mask, min_area=50, max_area=10000):
    """
    Fill in regions that are completely surrounded by the mask.
    This helps capture small islands within speech bubbles that should also be removed.
    
    Args:
        mask: Binary mask of speech bubbles (255 for bubble, 0 for non-bubble)
        min_area: Minimum area of enclosed region to fill
        max_area: Maximum area of enclosed region to fill
        
    Returns:
        Updated mask with enclosed regions filled
    """
    # Create a copy of the mask
    height, width = mask.shape
    filled_mask = mask.copy()
    
    # Invert the mask to find potential enclosed regions
    inverted = cv2.bitwise_not(mask)
    
    # Find all contours in the inverted mask
    contours, _ = cv2.findContours(inverted, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter out very large or small contours
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area <= area <= max_area:
            filtered_contours.append(contour)
    
    # For each contour, check if it's completely surrounded by the mask
    for contour in filtered_contours:
        # Create a mask for this contour
        contour_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], 0, 255, -1)
        
        # Dilate the contour slightly
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(contour_mask, kernel, iterations=1)
        
        # Subtract the original contour to get just the border
        border = cv2.subtract(dilated, contour_mask)
        
        # Check if this border is completely part of the original mask
        # If all border pixels are also in the original mask, then this region is enclosed
        overlap = cv2.bitwise_and(border, mask)
        border_pixels = np.sum(border > 0)
        overlap_pixels = np.sum(overlap > 0)
        
        # If 90% or more of the border is part of the mask, consider this enclosed
        if border_pixels > 0 and overlap_pixels / border_pixels > 0.9:
            # Fill this contour in the filled mask
            filled_mask = cv2.bitwise_or(filled_mask, contour_mask)
    
    # Safety check - don't let the filled mask cover more than 90% of the image
    filled_area = np.sum(filled_mask > 0)
    total_area = height * width
    
    if filled_area > total_area * 0.9:
        print("Warning: Fill enclosed regions created too large a mask, reverting to original mask")
        return mask
    
    return filled_mask

def process_image(input_path, output_path, method='transparency', draw_boxes=False):
    """
    Process a single image to remove speech bubbles.
    method: 'transparency', 'inpaint', or 'crop_only'
    """
    try:
        # Detect text regions
        boxes, img = detect_text_regions(input_path)
        
        # Also detect large white areas that might be speech bubbles
        whitespace_boxes = detect_speech_bubbles_with_whitespace(img)
        
        # Combine OCR and whitespace detection results
        all_boxes = boxes + whitespace_boxes
        
        if not all_boxes:
            # No text or bubbles detected, just copy the image but ensure it has alpha channel
            img_rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            cv2.imwrite(output_path, img_rgba)
            return False
        
        # Expand text regions to cover speech bubbles
        expanded_boxes, mask = expand_text_regions_to_bubble(img, all_boxes)
        
        # Fill in any regions completely surrounded by speech bubbles
        mask = fill_enclosed_regions(mask)
        
        # Safety check - make sure the mask doesn't cover more than 70% of the image
        height, width = img.shape[:2]
        panel_area = height * width
        mask_area = np.sum(mask > 0)
        max_mask_area = panel_area * 0.7
        
        if mask_area > max_mask_area:
            print(f"Warning: Mask for {input_path} covers too much area ({mask_area/panel_area:.1%}), skipping bubble removal")
            # Just convert to RGBA and save
            img_rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            cv2.imwrite(output_path, img_rgba)
            return False
        
        # Classify bubbles into top and interior
        top_bubbles, interior_bubbles = classify_speech_bubbles(img, expanded_boxes)
        
        # Create mask for interior bubbles only
        height, width = img.shape[:2]
        interior_mask = np.zeros((height, width), dtype=np.uint8)
        
        for box in interior_bubbles:
            x, y, w, h = box
            # Create more precise mask based on the mask from expand_text_regions_to_bubble
            roi_mask = mask[y:y+h, x:x+w]
            interior_mask[y:y+h, x:x+w] = roi_mask[0:h, 0:w]
        
        # Fill in enclosed regions in the interior mask as well
        interior_mask = fill_enclosed_regions(interior_mask)
        
        # Another safety check for interior mask
        interior_mask_area = np.sum(interior_mask > 0)
        if interior_mask_area > max_mask_area:
            print(f"Warning: Interior mask for {input_path} covers too much area ({interior_mask_area/panel_area:.1%}), skipping interior bubble processing")
            # Only do top bubble cropping if needed
            if top_bubbles:
                result = crop_top_bubbles(img, top_bubbles)
                result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
                cv2.imwrite(output_path, result)
                return True
            else:
                img_rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                cv2.imwrite(output_path, img_rgba)
                return False
        
        if draw_boxes:
            # Draw boxes around detected speech bubbles (for debugging)
            debug_img = img.copy()
            
            # Draw top bubbles in red
            for box in top_bubbles:
                x, y, w, h = box
                cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            
            # Draw interior bubbles in green
            for box in interior_bubbles:
                x, y, w, h = box
                cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Save debug image
            debug_path = output_path.replace('.png', '_debug.png')
            cv2.imwrite(debug_path, debug_img)
            
            # Also save a mask visualization to debug enclosed regions
            mask_viz = img.copy()
            # Overlay the mask in blue
            mask_viz[interior_mask > 0] = [255, 0, 0]  # Blue overlay for enclosed regions
            mask_debug_path = output_path.replace('.png', '_mask_debug.png')
            cv2.imwrite(mask_debug_path, mask_viz)
        
        # Process the image based on selected method
        if method == 'transparency':
            # Make interior bubbles transparent
            result = make_bubbles_transparent(img, interior_mask)
            # Then crop top bubbles if any
            if top_bubbles:
                result = crop_top_bubbles(result, top_bubbles)
                
        elif method == 'inpaint':
            # Inpaint interior bubbles
            if np.any(interior_mask > 0):
                result = inpaint_bubbles(img, interior_mask)
            else:
                result = img.copy()
                
            # Then crop top bubbles if any
            if top_bubbles:
                result = crop_top_bubbles(result, top_bubbles)
                
            # Convert to RGBA for consistency
            result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
            
        elif method == 'crop_only':
            # Only crop top bubbles, leave interior bubbles alone
            if top_bubbles:
                result = crop_top_bubbles(img, top_bubbles)
            else:
                result = img.copy()
                
            # Convert to RGBA for consistency
            result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
        
        # Save the result
        # Ensure output file has .png extension for alpha channel support
        output_path = output_path.replace('.jpg', '.png')
        cv2.imwrite(output_path, result)
        return True
        
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False

def process_directory(input_dir, output_dir, method='transparency', draw_boxes=False, limit=None):
    """
    Process all images in a directory.
    """
    print(f"Processing directory: {input_dir} -> {output_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}")
    
    # Get all PNG files in the input directory
    image_files = glob.glob(os.path.join(input_dir, "**", "*.png"), recursive=True)
    print(f"Found {len(image_files)} PNG files in {input_dir}")
    
    # Limit the number of files if specified
    if limit is not None and limit > 0:
        image_files = image_files[:limit]
        print(f"Limited to processing {len(image_files)} files")
    
    print(f"Will process {len(image_files)} images")
    
    processed_count = 0
    bubbles_count = 0
    
    # Process each image
    for img_path in tqdm(image_files, desc="Processing images"):
        print(f"Processing: {img_path}")
        
        # Create the same directory structure in the output directory
        rel_path = os.path.relpath(img_path, input_dir)
        out_path = os.path.join(output_dir, rel_path)
        print(f"Output path: {out_path}")
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        
        # Process the image
        has_bubbles = process_image(img_path, out_path, method, draw_boxes)
        
        processed_count += 1
        if has_bubbles:
            bubbles_count += 1
    
    print(f"Processed {processed_count} images")
    print(f"Found and processed speech bubbles in {bubbles_count} images")

def main():
    print("Starting enhanced speech bubble remover...")
    
    parser = argparse.ArgumentParser(description="Enhanced speech bubble remover for comic panels")
    parser.add_argument("--input", default="clean_panels", help="Input directory containing panel images")
    parser.add_argument("--output", default="panels_no_bubbles", help="Output directory for processed images")
    parser.add_argument("--method", choices=["transparency", "inpaint", "crop_only"], default="transparency",
                       help="Method to use for speech bubble removal (transparency, inpaint, or crop_only)")
    parser.add_argument("--debug", action="store_true", help="Draw boxes around detected speech bubbles")
    parser.add_argument("--limit", type=int, default=None, help="Limit processing to this many images (for testing)")
    
    args = parser.parse_args()
    print(f"Arguments: input={args.input}, output={args.output}, method={args.method}, debug={args.debug}, limit={args.limit}")
    
    start_time = time.time()
    
    # Check if input directory exists
    if not os.path.exists(args.input):
        print(f"Error: Input directory '{args.input}' does not exist")
        return
    
    # Check if there are image files in the input directory
    image_files = glob.glob(os.path.join(args.input, "**", "*.png"), recursive=True)
    if not image_files:
        print(f"Error: No PNG files found in '{args.input}'")
        return
    
    print(f"Found {len(image_files)} images in input directory")
    if args.limit:
        print(f"Processing only {args.limit} images (limit specified)")
    
    process_directory(args.input, args.output, args.method, args.debug, args.limit)
    
    elapsed_time = time.time() - start_time
    print(f"Total processing time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main() 