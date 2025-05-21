#!/usr/bin/env python3
"""
Panel Captioner (GPU-FORCED):
Automatically generate captions for Tintin comic panels using 
BLIP models on GPU with maximized performance.

Dependencies:
  pip install "transformers>=4.30.0" torch pillow pandas tqdm
"""

import os
import glob
import sys
import shutil
import argparse
from pathlib import Path
import time

# Prevent bitsandbytes from loading which causes errors
os.environ["BNB_DISABLE_CUDA_INIT"] = "1"

import torch  # torch.cuda.is_available()
from PIL import Image
import pandas as pd
from tqdm import tqdm

# Global default input directory
input_dir = "panels_no_bubbles"

def check_gpu():
    """Check for GPU availability and ensure it's properly configured"""
    try:
        # Force CUDA initialization
        torch.cuda.init()
        if not torch.cuda.is_available():
            print("CUDA not available! Checking why...")
            
            # Check for NVIDIA driver
            try:
                import subprocess
                result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
                if result.returncode != 0:
                    print("❌ nvidia-smi failed - NVIDIA drivers not installed or not working")
                else:
                    print(f"NVIDIA drivers found but CUDA still unavailable. Driver info:\n{result.stdout.splitlines()[0]}")
            except:
                print("❌ Could not check nvidia-smi - drivers likely not installed")
                
            print("⚠️ CUDA is supported but not available. Possible issues:")
            print("   - NVIDIA drivers not installed properly")
            print("   - Incompatible CUDA version")
            print("   - Running on a system without a compatible GPU")
            print("\n→ Defaulting to CPU (will be much slower)")
            return False
        
        # Force CUDA to initialize
        device_count = torch.cuda.device_count()
        print(f"✓ CUDA is available with {device_count} device(s):")
        for i in range(device_count):
            gpu_name = torch.cuda.get_device_name(i)
            total_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # Convert to GB
            print(f"  - GPU {i}: {gpu_name} ({total_mem:.2f} GB)")
        
        # Set default device to the first GPU
        torch.cuda.set_device(0)
        print(f"✓ Using GPU 0: {torch.cuda.get_device_name(0)}")
        
        # Create a small tensor on GPU to confirm it works
        x = torch.zeros(1).cuda()
        del x  # Free memory
        print("✓ Successfully created test tensor on GPU")
        
        return True
    except Exception as e:
        print(f"❌ Error initializing CUDA: {e}")
        return False

def load_blip_model(cache_dir=None, force_cpu=False):
    """Load BLIP model with forced GPU usage if available"""
    use_gpu = check_gpu() and not force_cpu
    device = "cuda" if use_gpu else "cpu"
    
    print(f"Loading BLIP model on {device.upper()}...")
    
    try:
        # Import BLIP
        from transformers import BlipProcessor, BlipForConditionalGeneration
        
        # Use base model first - more likely to fit on GPU
        model_name = "Salesforce/blip-image-captioning-base"
        print(f"Loading {model_name}...")
        
        # Settings for GPU loading
        kwargs = {}
        if cache_dir:
            kwargs["cache_dir"] = cache_dir
            
        # Load model components
        processor = BlipProcessor.from_pretrained(model_name, **kwargs)
        
        if use_gpu:
            print("Loading model directly to GPU...")
            model = BlipForConditionalGeneration.from_pretrained(model_name, **kwargs)
            
            # Force model to GPU
            model = model.cuda()
            
            # Verify it's on GPU
            device_str = next(model.parameters()).device
            if "cuda" not in str(device_str):
                print(f"⚠️ Model reports it's on {device_str}, forcing to CUDA...")
                model = model.cuda()
                
            # Check GPU memory usage
            mem_allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
            mem_reserved = torch.cuda.memory_reserved() / (1024**3)    # GB
            print(f"✓ Model loaded on GPU. Memory usage: {mem_allocated:.2f} GB (allocated) / {mem_reserved:.2f} GB (reserved)")
        else:
            # CPU loading
            model = BlipForConditionalGeneration.from_pretrained(model_name, **kwargs)
                
        print(f"✓ Successfully loaded {model_name}")
        return model, processor, device
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        
        if use_gpu:
            print("Trying again with CPU...")
            force_cpu = True
            return load_blip_model(cache_dir, force_cpu)
        else:
            print("Failed to load model. Check your transformers installation.")
            raise

def generate_caption(image_path, model, processor, device, style_suffix=True):
    """Generate a caption for a single image with optional style suffix"""
    try:
        # Open and prepare the image
        image = Image.open(image_path).convert("RGB")
        
        # Process image and force to device
        inputs = processor(image, return_tensors="pt")
        if device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items() if hasattr(v, "cuda")}
        
        # Generate caption
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=75)
            caption = processor.decode(outputs[0], skip_special_tokens=True)
        
        # Add style information if needed
        if style_suffix and not any(tag in caption for tag in ["Tintin", "Hergé", "ligne claire"]):
            caption += " Tintin comics, Hergé style, ligne claire illustration."
        
        return caption
    except Exception as e:
        print(f"❌ Error captioning {image_path}: {e}")
        return f"Error generating caption"

def save_caption(image_path, caption, output_dir=None):
    """Save caption to text file alongside the image"""
    try:
        rel = os.path.relpath(image_path, input_dir)
        base = os.path.splitext(rel)[0]
        target = os.path.join(output_dir or input_dir, f"{base}.txt")
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            f.write(caption)
    except Exception as e:
        print(f"❌ Error saving caption: {e}")

def process_images(in_dir, output_dir, csv_path, style_suffix, cache_dir, force_cpu, batch_size, sample_size=None):
    """Process all images in the input directory"""
    global input_dir
    input_dir = in_dir
    
    # Find all PNG files
    files = glob.glob(os.path.join(in_dir, "**", "*.png"), recursive=True)
    if not files:
        print(f"❌ No PNG files found in {in_dir}")
        return
    
    # Sample mode for testing
    if sample_size and sample_size > 0:
        if sample_size < len(files):
            from random import sample
            files = sample(files, sample_size)
            print(f"SAMPLE MODE: Processing {sample_size} random images for testing")
    
    print(f"Found {len(files)} images to process")
    
    # Load model
    try:
        model, processor, device = load_blip_model(cache_dir, force_cpu)
    except Exception as e:
        print(f"❌ Fatal error loading model: {e}")
        return
    
    # Process images
    results = []
    errors = 0
    start_time = time.time()
    
    for idx in range(0, len(files), batch_size):
        batch = files[idx:idx+batch_size]
        batch_results = []
        
        # Process each image in batch
        for img in tqdm(batch, desc=f"Batch {idx//batch_size+1}/{len(files)//batch_size+1}", leave=False):
            # Check GPU memory periodically
            if device == "cuda" and idx % 100 == 0:
                mem_allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
                print(f"GPU memory usage: {mem_allocated:.2f} GB")
                
            # Generate caption
            cap = generate_caption(img, model, processor, device, style_suffix)
            
            # Count errors
            if cap.startswith("Error"):
                errors += 1
                if errors > 10:
                    print("❌ Too many errors; aborting")
                    break
            
            # Save caption
            save_caption(img, cap, output_dir)
            batch_results.append({"filename": img, "caption": cap})
        
        # Add batch results to overall results
        results.extend(batch_results)
        
        # Show progress
        elapsed = time.time() - start_time
        processed = idx + len(batch)
        pct = int(processed / len(files) * 100)
        
        # Calculate ETA
        if processed > 0:
            seconds_per_image = elapsed / processed
            eta_seconds = seconds_per_image * (len(files) - processed)
            eta_min = int(eta_seconds / 60)
            eta_sec = int(eta_seconds % 60)
            print(f"Progress: {pct}% ({processed}/{len(files)}) - ETA: {eta_min}m {eta_sec}s")
            
            # Show time per image
            time_per_img = seconds_per_image
            if time_per_img > 60:
                print(f"  Average: {time_per_img/60:.1f} min/image")
            else:
                print(f"  Average: {time_per_img:.1f} sec/image")
        else:
            print(f"Progress: {pct}% ({processed}/{len(files)})")
        
        # Save partial CSV checkpoint
        if csv_path and (idx % (batch_size*10) == 0 or idx+batch_size >= len(files)):
            try:
                pd.DataFrame(results).to_csv(f"{csv_path}.partial", index=False)
            except Exception as e:
                print(f"❌ Error saving partial CSV: {e}")
                
        # Explicitly clear CUDA cache periodically to prevent memory issues
        if device == "cuda" and idx % 500 == 0:
            torch.cuda.empty_cache()
    
    # Final CSV output
    if csv_path:
        try:
            df = pd.DataFrame(results)
            df.to_csv(csv_path, index=False)
            if os.path.exists(f"{csv_path}.partial"):
                os.remove(f"{csv_path}.partial")
            print(f"✓ Saved captions CSV to {csv_path}")
        except Exception as e:
            print(f"❌ Error saving final CSV: {e}")
    
    # Report completion
    total_time = time.time() - start_time
    minutes = int(total_time / 60)
    seconds = int(total_time % 60)
    print(f"✓ Done: {len(results)} images in {minutes}m {seconds}s, {errors} errors")

def main():
    p = argparse.ArgumentParser(description="GPU-Forced Caption Generator")
    p.add_argument("--input",    default=input_dir, help="Image folder")
    p.add_argument("--output",   default=None,      help="Caption output folder")
    p.add_argument("--csv",      default="captions.csv", help="CSV path")
    p.add_argument("--no-style", action="store_true", help="Skip style suffix")
    p.add_argument("--cache-dir",default=None,      help="HF cache dir")
    p.add_argument("--clear-cache", action="store_true", help="Clear cache first")
    p.add_argument("--batch-size", type=int, default=1, help="Images per batch")
    p.add_argument("--force-cpu", action="store_true", help="Force CPU usage")
    p.add_argument("--sample", type=int, default=0, help="Process only N random images (for testing)")
    args = p.parse_args()
    
    # Validate input directory
    if not os.path.isdir(args.input):
        print(f"❌ Input directory '{args.input}' does not exist")
        return
    
    # Clear cache if requested
    if args.clear_cache and args.cache_dir:
        print(f"Clearing cache at {args.cache_dir}...")
        shutil.rmtree(args.cache_dir, ignore_errors=True)
        os.makedirs(args.cache_dir, exist_ok=True)
    
    # Process images
    process_images(
        args.input,
        args.output,
        args.csv,
        not args.no_style,
        args.cache_dir,
        args.force_cpu,
        args.batch_size,
        args.sample
    )

if __name__ == "__main__":
    main()
