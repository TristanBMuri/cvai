#!/usr/bin/env python3
"""
Panel Extractor: A tool to automatically detect and crop individual comic panels from PDF pages.

Dependencies:
  - pdf2image (for converting PDF pages to images)
  - poppler-utils (external; required by pdf2image)
  - opencv-python
  - numpy

Install:
  pip install pdf2image opencv-python numpy
  
For Windows, you also need to:
  1. Download poppler for Windows from: https://github.com/oschwartz10612/poppler-windows/releases/
  2. Extract the ZIP file
  3. Add the extracted bin folder to your PATH or specify the path in the script
"""
import os
import cv2
import numpy as np
from pdf2image import convert_from_path
import sys

def extract_panels_from_image(img, min_area=10000):
    """Extract comic panels from an image using combined contour and grid detection."""
    # Get image dimensions
    height, width = img.shape[:2]
    page_area = height * width
    
    # First try contour detection for natural panel boundaries
    contour_panels = detect_panels_by_contours(img)
    
    # If contour detection found a reasonable number of panels, use that
    if len(contour_panels) >= 4 and len(contour_panels) <= 16:
        print(f"Using contour detection: {len(contour_panels)} panels")
        return contour_panels
    
    # Otherwise, try content-aware grid detection
    adaptive_panels = detect_adaptive_grid(img)
    if adaptive_panels:
        print(f"Using adaptive grid: {len(adaptive_panels)} panels")
        return adaptive_panels
    
    # Finally, fall back to standard grid as last resort
    grid_panels = detect_basic_grid(img)
    print(f"Using basic grid: {len(grid_panels)} panels")
    return grid_panels

def detect_panels_by_contours(img):
    """Detect panels using contour detection - allows for natural panel boundaries."""
    height, width = img.shape[:2]
    page_area = height * width
    
    # Convert to gray and apply preprocessing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter to reduce noise while preserving edges
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 11, 2)
    
    # Apply morphological operations to enhance panel borders
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Apply dilation to connect panel borders
    kernel_dil = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(closed, kernel_dil, iterations=1)
    
    # Find contours of panel regions
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Calculate minimum and maximum panel sizes based on page size
    min_panel_area = page_area * 0.01  # At least 1% of page
    max_panel_area = page_area * 0.6   # At most 60% of page
    
    # Process contours
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        
        # Filter by area
        if area < min_panel_area or area > max_panel_area:
            continue
            
        # Filter by aspect ratio
        aspect_ratio = w / float(h)
        if aspect_ratio > 5 or aspect_ratio < 0.2:
            continue
            
        # Check if panel contains actual content (not just white space)
        roi = gray[y:y+h, x:x+w]
        if roi.size == 0 or np.std(roi) < 20:  # Skip low contrast regions
            continue
            
        # Add padding for better panel cropping
        padding = min(int(min(w, h) * 0.05), 15)  # 5% padding or 15px max
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(width - x, w + 2*padding)
        h = min(height - y, h + 2*padding)
        
        boxes.append((x, y, w, h))
    
    # Sort boxes top-to-bottom, left-to-right
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    
    # Convert boxes to panel images
    panels = []
    for x, y, w, h in boxes:
        # Skip if panel is too small
        if w < 50 or h < 50:
            continue
            
        panel_img = img[y:y+h, x:x+w].copy()
        panels.append(panel_img)
    
    return panels

def detect_adaptive_grid(img):
    """Detect panels using an adaptive grid based on edge detection."""
    height, width = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Find horizontal and vertical lines using morphological operations
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    
    horizontal = cv2.morphologyEx(edges, cv2.MORPH_OPEN, h_kernel)
    vertical = cv2.morphologyEx(edges, cv2.MORPH_OPEN, v_kernel)
    
    # Dilate to connect broken lines
    h_dilated = cv2.dilate(horizontal, h_kernel, iterations=1)
    v_dilated = cv2.dilate(vertical, v_kernel, iterations=1)
    
    # Create horizontal and vertical projections
    h_proj = np.sum(v_dilated, axis=0) / height  # Vertical lines create horizontal divisions
    v_proj = np.sum(h_dilated, axis=1) / width   # Horizontal lines create vertical divisions
    
    # Smooth projections
    h_proj = np.convolve(h_proj, np.ones(20)/20, mode='same')
    v_proj = np.convolve(v_proj, np.ones(20)/20, mode='same')
    
    # Find peaks in projections (potential panel boundaries)
    h_peaks = find_peaks(h_proj, height_factor=1.5)
    v_peaks = find_peaks(v_proj, height_factor=1.5)
    
    # Only continue if we found reasonable grid lines
    if len(h_peaks) < 1 or len(v_peaks) < 1:
        return []
        
    # Add page boundaries
    h_divs = [0] + sorted(h_peaks) + [width]
    v_divs = [0] + sorted(v_peaks) + [height]
    
    # Create panels from grid intersections
    panel_imgs = []
    
    # Generate all possible grid cells
    for i in range(len(v_divs) - 1):
        for j in range(len(h_divs) - 1):
            y1 = v_divs[i]
            y2 = v_divs[i + 1]
            x1 = h_divs[j]
            x2 = h_divs[j + 1]
            
            # Skip if cell is too small
            if (x2 - x1) < 80 or (y2 - y1) < 80:
                continue
                
            # Validate cell contains content
            roi = gray[y1:y2, x1:x2]
            if roi.size == 0:
                continue
                
            std_dev = np.std(roi)
            mean_val = np.mean(roi)
            
            # Skip cells with little variation (likely empty)
            if std_dev < 25 or (mean_val > 240 and std_dev < 30):
                continue
                
            # Add valid panel
            panel = img[y1:y2, x1:x2].copy()
            panel_imgs.append(panel)
    
    return panel_imgs

def detect_basic_grid(img):
    """Create a basic grid of panels as fallback."""
    height, width = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Try a few common comic grid layouts
    grids = [
        (3, 3),  # 9-panel grid
        (3, 2),  # 6-panel grid
        (2, 2),  # 4-panel grid
        (4, 2),  # 8-panel grid
    ]
    
    best_panels = []
    best_count = 0
    
    for rows, cols in grids:
        # Calculate panel dimensions with margins
        margin = int(min(width, height) * 0.05)  # 5% margin
        panel_width = (width - margin*(cols+1)) // cols
        panel_height = (height - margin*(rows+1)) // rows
        
        panels = []
        
        for r in range(rows):
            for c in range(cols):
                x = margin + c * (panel_width + margin)
                y = margin + r * (panel_height + margin)
                
                # Skip if out of bounds
                if x + panel_width > width or y + panel_height > height:
                    continue
                
                # Check if panel has content
                roi = gray[y:y+panel_height, x:x+panel_width]
                std_dev = np.std(roi)
                
                if std_dev < 20:  # Skip low contrast cells
                    continue
                
                # Add valid panel
                panel = img[y:y+panel_height, x:x+panel_width].copy()
                panels.append(panel)
        
        # Keep layout with most valid panels
        if len(panels) > best_count:
            best_panels = panels
            best_count = len(panels)
    
    return best_panels

def find_peaks(projection, height_factor=1.5):
    """Find peaks in a projection (areas with high values)."""
    # Normalize projection
    if np.max(projection) > 0:
        norm_proj = projection / np.max(projection)
    else:
        return []
    
    # Calculate threshold based on median value
    threshold = np.median(norm_proj) * height_factor
    
    # Find peaks
    peaks = []
    above_threshold = False
    start_idx = 0
    
    for i in range(len(norm_proj)):
        if norm_proj[i] > threshold and not above_threshold:
            above_threshold = True
            start_idx = i
        elif norm_proj[i] <= threshold and above_threshold:
            above_threshold = False
            # Add middle of peak
            peaks.append((start_idx + i) // 2)
    
    return peaks

def process_pdf(pdf_path, output_dir, dpi=200, min_area=10000, poppler_path=None):
    """Process a single PDF file and extract panels."""
    try:
        if poppler_path:
            pages = convert_from_path(pdf_path, dpi=dpi, poppler_path=poppler_path)
        else:
            pages = convert_from_path(pdf_path, dpi=dpi)
    except Exception as e:
        if "poppler" in str(e).lower():
            print("\nERROR: Poppler not found. Please install poppler-utils:")
            print("1. Download from: https://github.com/oschwartz10612/poppler-windows/releases/")
            print("2. Extract the ZIP file")
            print("3. Add the bin folder to your PATH or specify the path in the script")
            print("\nAlternatively, set the poppler_path parameter to the folder containing the poppler binaries")
            return
        else:
            raise e
            
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    out_sub = os.path.join(output_dir, base)
    os.makedirs(out_sub, exist_ok=True)
    
    print(f"Processing {base} ({len(pages)} pages)...")
    
    panel_count = 0
    for idx, page in enumerate(pages, start=1):
        # Convert PIL image to OpenCV format
        page_img = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
        
        # For the first page (title page), preserve it as a single image without panels
        if idx == 1:
            fname = f"{base}_title_page.png"
            cv2.imwrite(os.path.join(out_sub, fname), page_img)
            print(f"  Page {idx}/{len(pages)} - Saved as title page")
            panel_count += 1
            continue
        
        # For all other pages, detect panels
        panels = extract_panels_from_image(page_img, min_area=min_area)
        
        print(f"  Page {idx}/{len(pages)} - Found {len(panels)} panels")
        panel_count += len(panels)
        
        for j, panel in enumerate(panels, start=1):
            fname = f"{base}_page{idx:02d}_panel{j:02d}.png"
            cv2.imwrite(os.path.join(out_sub, fname), panel)
    
    print(f"  Total panels extracted: {panel_count}")


def main():
    # Hardcoded paths
    input_dir = r"C:\Code\Tintin\TinTin -- Full PDF Collector's Edition\Relevant\cleaned_pdfs"
    output_dir = r"C:\Code\Tintin\TinTin -- Full PDF Collector's Edition\Relevant\panels"
    dpi = 200
    min_area = 10000
    
    # Try different potential poppler paths
    poppler_paths = [
        r"C:\Users\leona\Downloads\poppler\poppler-24.08.0\Library\bin",
        r"C:\Users\leona\Downloads\Release-24.08.0-0\poppler-24.08.0\bin",
        r"C:\Users\leona\Downloads\poppler-24.08.0\bin",
        r"C:\Users\leona\Downloads\Release-24.08.0-0\bin",
        # Add the directory containing pdfinfo.exe and pdftoppm.exe
        # For example:
    ]
    
    # Check if any of the paths contain the necessary binaries
    poppler_path = None
    for path in poppler_paths:
        if os.path.exists(path):
            print(f"Found potential poppler path: {path}")
            if os.path.exists(os.path.join(path, "pdfinfo.exe")) and os.path.exists(os.path.join(path, "pdftoppm.exe")):
                print(f"Confirmed valid poppler path: {path}")
                poppler_path = path
                break
            else:
                print(f"Path exists but missing required binaries: {path}")
    
    if not poppler_path:
        print("\nERROR: Could not find valid poppler path. Please:")
        print("1. Extract the ZIP file from: C:\\Users\\leona\\Downloads\\Release-24.08.0-0.zip")
        print("2. Locate the folder containing pdfinfo.exe and pdftoppm.exe")
        print("3. Update the script with the correct path")
        return
    
    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist")
        print(f"Please create it and add your PDF files there")
        return
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get all PDFs in the input directory
    pdfs = [os.path.join(input_dir, f)
            for f in os.listdir(input_dir)
            if f.lower().endswith('.pdf')]
    
    if not pdfs:
        print(f"No PDF files found in '{input_dir}'")
        print(f"Please add your PDF files to this directory")
        return
        
    print(f"Found {len(pdfs)} PDF files to process")

    # Process each PDF
    for pdf in pdfs:
        try:
            process_pdf(pdf, output_dir, dpi=dpi, min_area=min_area, poppler_path=poppler_path)
        except Exception as e:
            print(f"Error processing {pdf}: {e}")
            import traceback
            traceback.print_exc()
    
    print("All done!")

if __name__ == '__main__':
    main() 