#!/usr/bin/env python3
"""
Utility to merge split safetensors files back into single files.
Used during container startup to merge model chunks.
"""
import os
import glob

def main():
    """Merge .safetensors.part-* files back into .safetensors files."""
    models_dir = os.environ.get("MODELS_DIR", "/app/models")
    
    if not os.path.exists(models_dir):
        return
    
    merged_count = 0
    for root, dirs, files in os.walk(models_dir):
        # Find all .part-* files
        part_files = sorted(glob.glob(os.path.join(root, "*.safetensors.part-*")))
        
        if not part_files:
            continue
        
        # Group by base filename
        base_files = {}
        for part_file in part_files:
            # Extract base filename (remove .part-*)
            base_name = part_file.rsplit(".part-", 1)[0]
            if base_name not in base_files:
                base_files[base_name] = []
            base_files[base_name].append(part_file)
        
        # Merge each group
        for base_name, parts in base_files.items():
            if os.path.exists(base_name):
                # Already merged
                continue
            
            print(f"Merging {len(parts)} parts into {base_name}...")
            with open(base_name, "wb") as outfile:
                for part in sorted(parts):
                    with open(part, "rb") as infile:
                        outfile.write(infile.read())
            
            # Remove part files
            for part in parts:
                try:
                    os.remove(part)
                except:
                    pass
            
            merged_count += 1
            print(f"✓ Merged {base_name}")
    
    if merged_count > 0:
        print(f"✓ Merged {merged_count} safetensors files")
    else:
        print("No safetensors parts to merge")

if __name__ == "__main__":
    main()
