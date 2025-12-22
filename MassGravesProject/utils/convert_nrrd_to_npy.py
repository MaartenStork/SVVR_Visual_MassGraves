#!/usr/bin/env python3
"""
Convert NRRD segmentation file to .npy + header.json + labels.json format.
Extracts segment labels and colors from the NRRD metadata.
"""

import numpy as np
import nrrd
import json
from pathlib import Path
import re


def convert_nrrd_segmentation(nrrd_path, output_prefix=None):
    """
    Convert NRRD segmentation to numpy array + metadata files.
    
    Args:
        nrrd_path: Path to the .nrrd file
        output_prefix: Prefix for output files (default: based on input name)
    """
    nrrd_path = Path(nrrd_path)
    
    if output_prefix is None:
        output_prefix = nrrd_path.stem.replace(' ', '_')
    
    output_dir = nrrd_path.parent.parent  # Go up to 2021.003 folder
    
    print("="*70)
    print(f"Converting: {nrrd_path.name}")
    print("="*70)
    
    # Read the NRRD file
    print("\nReading NRRD file...")
    data, header = nrrd.read(str(nrrd_path))
    
    print(f"Data shape: {data.shape}")
    print(f"Data type: {data.dtype}")
    print(f"Unique labels: {len(np.unique(data))} (including background)")
    
    # Extract header info for alignment
    header_info = {
        "type": header.get('type', 'unsigned char'),
        "dimension": header.get('dimension', 3),
        "sizes": list(data.shape),
        "space directions": [list(d) for d in header['space directions']],
        "encoding": "raw",  # We'll save as raw numpy
        "space origin": list(header['space origin'])
    }
    
    # Extract segment information from header
    segments = {}
    segment_pattern = re.compile(r'Segment(\d+)_(\w+)')
    
    for key, value in header.items():
        match = segment_pattern.match(key)
        if match:
            seg_idx = match.group(1)
            prop_name = match.group(2)
            
            if seg_idx not in segments:
                segments[seg_idx] = {}
            
            segments[seg_idx][prop_name] = value
    
    # Build label mapping: label_value -> name, color
    label_info = {}
    for seg_idx, props in segments.items():
        if 'LabelValue' in props and 'Name' in props:
            label_value = int(props['LabelValue'])
            name = props['Name']
            
            # Parse color (format: "0.615686 0.423529 0.635294")
            color = [1.0, 1.0, 1.0]  # Default white
            if 'Color' in props:
                try:
                    color = [float(c) for c in props['Color'].split()]
                except:
                    pass
            
            label_info[label_value] = {
                "name": name,
                "color": color,
                "id": props.get('ID', name.lower().replace(' ', '_'))
            }
    
    print(f"\nFound {len(label_info)} labeled segments:")
    
    # Group by category for display
    bones = []
    organs = []
    other = []
    
    for label_val, info in sorted(label_info.items()):
        name = info['name'].lower()
        if any(b in name for b in ['rib', 'vertebra', 'sacrum', 'hip', 'femur', 
                                    'scapula', 'clavicula', 'humerus', 'sternum',
                                    'costal', 'iliac']):
            bones.append((label_val, info['name']))
        elif any(o in name for o in ['spleen', 'kidney', 'liver', 'lung', 'heart',
                                      'aorta', 'stomach', 'pancreas', 'gallbladder',
                                      'adrenal', 'intestine', 'colon', 'duodenum',
                                      'esophagus', 'trachea', 'bladder']):
            organs.append((label_val, info['name']))
        else:
            other.append((label_val, info['name']))
    
    if bones:
        print(f"\n  Bones ({len(bones)}):")
        for lv, name in bones[:10]:
            print(f"    Label {lv}: {name}")
        if len(bones) > 10:
            print(f"    ... and {len(bones)-10} more bones")
    
    if organs:
        print(f"\n  Organs ({len(organs)}):")
        for lv, name in organs[:5]:
            print(f"    Label {lv}: {name}")
        if len(organs) > 5:
            print(f"    ... and {len(organs)-5} more organs")
    
    if other:
        print(f"\n  Other ({len(other)}):")
        for lv, name in other[:3]:
            print(f"    Label {lv}: {name}")
        if len(other) > 3:
            print(f"    ... and {len(other)-3} more")
    
    # Save files
    output_npy = output_dir / f"{output_prefix}_segmentation.npy"
    output_header = output_dir / f"{output_prefix}_header.json"
    output_labels = output_dir / f"{output_prefix}_labels.json"
    
    print(f"\nSaving files to {output_dir}:")
    
    # Save numpy array
    np.save(output_npy, data)
    print(f"  ✓ {output_npy.name} ({data.nbytes / 1024 / 1024:.1f} MB)")
    
    # Save header
    with open(output_header, 'w') as f:
        json.dump(header_info, f, indent=2)
    print(f"  ✓ {output_header.name}")
    
    # Save labels
    with open(output_labels, 'w') as f:
        json.dump(label_info, f, indent=2)
    print(f"  ✓ {output_labels.name}")
    
    print("\n" + "="*70)
    print("✓ Conversion complete!")
    print("="*70)
    
    return output_npy, output_header, output_labels


if __name__ == "__main__":
    import sys
    
    script_dir = Path(__file__).parent
    
    # Default: look for the bone segmentation file
    nrrd_file = script_dir / "DICOM_upper_body" / "3 CAP w-o  5.0  B31f segmentation.nrrd"
    
    if len(sys.argv) > 1:
        nrrd_file = Path(sys.argv[1])
    
    if not nrrd_file.exists():
        print(f"Error: NRRD file not found: {nrrd_file}")
        sys.exit(1)
    
    convert_nrrd_segmentation(nrrd_file, output_prefix="upper_body_bones")

