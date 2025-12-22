#!/usr/bin/env python3
"""
Extract lower body DICOM files to a separate folder for 3D Slicer processing.
Analyzes SeriesNumber and Z-position to identify and separate leg/lower body scans.
"""

import os
import shutil
from pathlib import Path
from collections import defaultdict
import pydicom

def analyze_and_extract_lower_body():
    script_dir = Path(__file__).parent
    dicom_dir = script_dir / "DICOM"
    
    if not dicom_dir.exists():
        print(f"Error: DICOM directory not found at {dicom_dir}")
        return
    
    print("="*70)
    print("Analyzing DICOM series to find lower body scan...")
    print("="*70)
    
    # Group files by SeriesNumber
    series_data = defaultdict(list)
    
    print("\nScanning DICOM files...")
    files = list(dicom_dir.glob('*'))
    total = len(files)
    
    for i, file_path in enumerate(files):
        if file_path.is_file():
            try:
                ds = pydicom.dcmread(str(file_path), stop_before_pixels=True)
                if hasattr(ds, 'SeriesNumber') and hasattr(ds, 'ImagePositionPatient'):
                    series_num = ds.SeriesNumber
                    z_pos = float(ds.ImagePositionPatient[2])
                    series_desc = ds.SeriesDescription if hasattr(ds, 'SeriesDescription') else 'N/A'
                    pixel_spacing = ds.PixelSpacing if hasattr(ds, 'PixelSpacing') else [1, 1]
                    
                    series_data[series_num].append({
                        'file': file_path,
                        'z_pos': z_pos,
                        'desc': series_desc,
                        'spacing': pixel_spacing
                    })
            except Exception as e:
                pass
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{total} files...")
    
    print(f"\nFound {len(series_data)} different series:")
    print("-"*70)
    
    series_info = []
    for series_num, items in series_data.items():
        if len(items) > 5:  # Only consider series with more than 5 slices
            z_positions = [item['z_pos'] for item in items]
            min_z = min(z_positions)
            max_z = max(z_positions)
            desc = items[0]['desc']
            spacing = items[0]['spacing']
            
            series_info.append({
                'series_num': series_num,
                'count': len(items),
                'min_z': min_z,
                'max_z': max_z,
                'desc': desc,
                'spacing': spacing,
                'items': items
            })
            
            print(f"  Series {series_num}: {len(items)} slices")
            print(f"    Z range: {min_z:.1f} to {max_z:.1f} mm")
            print(f"    Description: {desc}")
            print(f"    Pixel spacing: {float(spacing[0]):.3f} mm")
            print()
    
    if len(series_info) < 2:
        print("Only one series found - cannot separate upper/lower body")
        return
    
    # Sort by min Z position (lower Z = feet, higher Z = head in DICOM)
    series_info.sort(key=lambda x: x['min_z'])
    
    print("="*70)
    print("Series sorted by Z position (feet to head):")
    print("="*70)
    for i, s in enumerate(series_info):
        position = "LOWER BODY (feet/legs)" if i == 0 else "UPPER BODY (torso/head)" if i == len(series_info)-1 else "MIDDLE"
        print(f"  {i+1}. Series {s['series_num']}: Z={s['min_z']:.1f} to {s['max_z']:.1f} → {position}")
    
    # The series with the LOWEST Z values is the lower body (feet/legs)
    lower_body_series = series_info[0]
    
    print("\n" + "="*70)
    print(f"LOWER BODY identified: Series {lower_body_series['series_num']}")
    print(f"  {lower_body_series['count']} slices")
    print(f"  Z range: {lower_body_series['min_z']:.1f} to {lower_body_series['max_z']:.1f} mm")
    print("="*70)
    
    # Create output directory
    output_dir = script_dir / "DICOM_lower_body"
    if output_dir.exists():
        print(f"\nOutput directory already exists: {output_dir}")
        response = input("Delete and recreate? (y/n): ").strip().lower()
        if response == 'y':
            shutil.rmtree(output_dir)
        else:
            print("Aborted.")
            return
    
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nCopying {lower_body_series['count']} files to {output_dir}...")
    
    for i, item in enumerate(lower_body_series['items']):
        src = item['file']
        dst = output_dir / src.name
        shutil.copy2(src, dst)
        
        if (i + 1) % 50 == 0:
            print(f"  Copied {i+1}/{lower_body_series['count']} files...")
    
    print("\n" + "="*70)
    print("✓ DONE!")
    print("="*70)
    print(f"Lower body DICOM files copied to: {output_dir}")
    print(f"Total files: {lower_body_series['count']}")
    print("\nYou can now load this folder in 3D Slicer for segmentation.")
    
    # Also offer to create upper body folder
    print("\n" + "-"*70)
    response = input("Also create DICOM_upper_body folder? (y/n): ").strip().lower()
    if response == 'y':
        upper_body_series = series_info[-1]  # Highest Z = upper body
        upper_dir = script_dir / "DICOM_upper_body"
        
        if upper_dir.exists():
            shutil.rmtree(upper_dir)
        upper_dir.mkdir()
        
        print(f"\nCopying {upper_body_series['count']} upper body files...")
        for item in upper_body_series['items']:
            src = item['file']
            dst = upper_dir / src.name
            shutil.copy2(src, dst)
        
        print(f"✓ Upper body files copied to: {upper_dir}")


if __name__ == "__main__":
    analyze_and_extract_lower_body()

