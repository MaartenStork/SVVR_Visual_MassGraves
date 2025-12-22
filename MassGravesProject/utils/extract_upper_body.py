#!/usr/bin/env python3
"""
Extract upper body DICOM files (Series 5 - torso/head) to a separate folder.
"""

import os
import shutil
from pathlib import Path
from collections import defaultdict
import pydicom

def extract_upper_body():
    script_dir = Path(__file__).parent
    dicom_dir = script_dir / "DICOM"
    output_dir = script_dir / "DICOM_upper_body"
    
    # Target: Series 5 (214 slices, Z=-36.5 to 1028.5 mm - main torso/head)
    target_series = 5
    
    print("="*70)
    print(f"Extracting Series {target_series} (upper body/torso) to separate folder...")
    print("="*70)
    
    if output_dir.exists():
        print(f"Removing existing folder: {output_dir}")
        shutil.rmtree(output_dir)
    
    output_dir.mkdir()
    
    files = list(dicom_dir.glob('*'))
    copied = 0
    
    print(f"Scanning {len(files)} DICOM files...")
    
    for file_path in files:
        if file_path.is_file():
            try:
                ds = pydicom.dcmread(str(file_path), stop_before_pixels=True)
                if hasattr(ds, 'SeriesNumber') and ds.SeriesNumber == target_series:
                    dst = output_dir / file_path.name
                    shutil.copy2(file_path, dst)
                    copied += 1
            except:
                pass
    
    print(f"\n✓ Copied {copied} files to {output_dir}")
    print("This is the main torso/head scan (Z=-36.5 to 1028.5 mm)")


if __name__ == "__main__":
    extract_upper_body()

