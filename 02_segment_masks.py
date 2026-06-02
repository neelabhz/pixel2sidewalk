#!/usr/bin/env python3
import sys
import os
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import json
import argparse
import torch

# Reduce fragmentation on large allocations
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Import logic from Testing/segment_sidewalks.py
sys.path.insert(0, "./")
from Testing.segment_sidewalks import load_segmentor, segment_image_full_resolution

def main():
    parser = argparse.ArgumentParser(description="Segment section images in a flat directory")
    parser.add_argument("section_dir", type=Path, help="Section directory")
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    
    section_dir = args.section_dir
    images_dir = section_dir / "images"
    output_dir = section_dir / "segmentation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Only process non-equi normal images
    image_files = [f for f in sorted(images_dir.glob("*.jpg")) if not f.name.endswith(".equi.jpg")]
    if not image_files:
        print(f"No regular JPG images found in {images_dir}")
        return
        
    print(f"Processing {len(image_files)} images from {section_dir.name} using DINOv3")
    
    model = load_segmentor(device="cuda")
    
    for img_path in tqdm(image_files, desc=f"Segmenting {section_dir.name}"):
        
        # Output exactly the name format that sidewalk_project_section.py expects
        mask_path = output_dir / f"{img_path.stem}_mask.png"
        
        if mask_path.exists():
            continue
            
        try:
            # 1. Check if image is excessively large
            orig_img = Image.open(img_path)
            orig_w, orig_h = orig_img.size
            # Cap panoramic images at 2048px max dimension to prevent OOM
            max_dim = 2048
            
            target_path = img_path
            temp_path = section_dir / f"temp_{img_path.name}"
            
            needs_resize = (orig_w > max_dim or orig_h > max_dim)
            if needs_resize:
                # Downsample to speed up inference and prevent OOM
                scale = max_dim / max(orig_w, orig_h)
                new_w, new_h = int(orig_w * scale), int(orig_h * scale)
                resized_img = orig_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                resized_img.convert("RGB").save(temp_path)
                target_path = temp_path
            
            # Use batch_size=1 for large images to reduce VRAM pressure
            effective_batch = 1 if (orig_w * orig_h > 4_000_000) else args.batch_size
            
            # 2. Run segmentation (on original or downsampled)
            sidewalk_mask, _ = segment_image_full_resolution(
                model, target_path, device="cuda",
                tile_size=1024, overlap=128, max_single=2048, batch_size=effective_batch
            )
            
            # 3. Upscale mask back to original resolution if it was downsampled
            mask_img = Image.fromarray((sidewalk_mask * 255).astype(np.uint8))
            if needs_resize:
                mask_img = mask_img.resize((orig_w, orig_h), Image.Resampling.NEAREST)
                if temp_path.exists():
                    temp_path.unlink()  # Clean up temp file
                    
            # 4. Save final mask
            mask_img.save(mask_path)
            
        except Exception as e:
            if 'temp_path' in locals() and temp_path.exists():
                temp_path.unlink()
            print(f"Error processing {img_path.name}: {e}")
        finally:
            # Clear CUDA cache between images to prevent memory fragmentation
            torch.cuda.empty_cache()
            
    print(f"Segmentation complete for {section_dir.name}")

if __name__ == "__main__":
    main()
