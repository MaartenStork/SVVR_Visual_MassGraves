#!/usr/bin/env python3
"""
DICOM Viewer with Aorta Segmentation
Shows CT scan with red aorta overlay in VTK
"""

import sys
from pathlib import Path
from dicom_viewer_3d import DICOM3DViewer

def main():
    # Get the DICOM directory
    script_dir = Path(__file__).parent
    dicom_dir = script_dir / "DICOM"
    
    if not dicom_dir.exists():
        print(f"Error: DICOM directory not found at {dicom_dir}")
        sys.exit(1)
    
    # Check for aorta segmentation - prefer fullbody version
    aorta_seg_path = script_dir / "fullbody_aorta_segmentation.npy"
    if not aorta_seg_path.exists():
        # Fallback to old segmentation
        aorta_seg_path = script_dir / "aorta_segmentation_0.npy"
    if not aorta_seg_path.exists():
        aorta_seg_path = None
        print("Note: Aorta segmentation not found, will display CT only")
    
    # Create viewer
    print("="*60)
    print("Loading CT scan with Aorta segmentation...")
    print("="*60)
    viewer = DICOM3DViewer(dicom_dir, aorta_seg_path=aorta_seg_path)
    
    print("\n" + "="*60)
    print("Starting VTK Volume Rendering with Aorta (RED)")
    print("="*60)
    print("Controls:")
    print("  - Left click + drag: Rotate")
    print("  - Middle click + drag: Pan")
    print("  - Right click + drag: Zoom")
    print("  - Scroll wheel: Zoom")
    print("="*60 + "\n")
    
    # Show VTK volume with aorta
    viewer.show_vtk_volume()

if __name__ == "__main__":
    main()

