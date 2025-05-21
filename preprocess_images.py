import os
import shutil
from pathlib import Path
from PIL import Image
import numpy as np

def preprocess_image(img_path, target_size=512):
    """
    Preprocess an image for LoRA training:
    1. Load RGBA image
    2. Extract alpha channel for mask
    3. Composite RGB onto white background
    4. Resize to target size
    5. Return processed image and mask
    """
    # Load image
    img = Image.open(img_path).convert('RGBA')
    
    # Split into RGB and alpha
    r, g, b, a = img.split()
    
    # Create white background
    white_bg = Image.new('RGB', img.size, (255, 255, 255))
    
    # Composite RGB onto white background using alpha
    rgb_img = Image.alpha_composite(white_bg.convert('RGBA'), img).convert('RGB')
    
    # Calculate padding
    width, height = rgb_img.size
    max_dim = max(width, height)
    pad_width = (max_dim - width) // 2
    pad_height = (max_dim - height) // 2
    
    # Create padded image
    padded_img = Image.new('RGB', (max_dim, max_dim), (255, 255, 255))
    padded_img.paste(rgb_img, (pad_width, pad_height))
    
    # Resize to target size
    resized_img = padded_img.resize((target_size, target_size), Image.LANCZOS)
    
    # Process mask
    mask = Image.fromarray(np.array(a) > 0)  # Convert to binary mask
    padded_mask = Image.new('L', (max_dim, max_dim), 0)
    padded_mask.paste(mask, (pad_width, pad_height))
    resized_mask = padded_mask.resize((target_size, target_size), Image.NEAREST)
    
    return resized_img, resized_mask

def preprocess_dataset(source_dir, target_dir, target_size=512):
    """
    Preprocess all images in the source directory and save to target directory.
    """
    # Create target directories
    target_path = Path(target_dir)
    target_path.mkdir(exist_ok=True, parents=True)
    
    # Create masks directory
    masks_path = target_path / "masks"
    masks_path.mkdir(exist_ok=True, parents=True)
    
    # Counter for sequential naming
    counter = 1
    
    # Process all PNG files
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.png'):
                # Get paths
                img_path = os.path.join(root, file)
                caption_path = f"{os.path.splitext(img_path)[0]}.txt"
                
                # Skip if no caption
                if not os.path.exists(caption_path):
                    print(f"Warning: No caption found for {img_path}")
                    continue
                
                try:
                    # Process image
                    processed_img, mask = preprocess_image(img_path, target_size)
                    
                    # Create new filenames
                    new_img_name = f"image_{counter:04d}.png"
                    new_mask_name = f"image_{counter:04d}_mask.png"
                    new_caption_name = f"image_{counter:04d}.txt"
                    
                    # Save processed files
                    processed_img.save(target_path / new_img_name)
                    mask.save(masks_path / new_mask_name)
                    shutil.copy2(caption_path, target_path / new_caption_name)
                    
                    print(f"Processed {img_path} -> {new_img_name}")
                    counter += 1
                    
                except Exception as e:
                    print(f"Error processing {img_path}: {str(e)}")
    
    print(f"Preprocessed {counter-1} images in {target_dir}")
    print(f"Masks saved in {masks_path}")

if __name__ == "__main__":
    source_directory = "panels_no_bubbles"  # Source directory with original images
    target_directory = "processed_dataset"  # Directory for processed images
    
    preprocess_dataset(source_directory, target_directory) 