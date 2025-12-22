#!/usr/bin/env python3
"""
Quick launcher for DICOM 3D viewer
Shows all visualization modes automatically
Supports multiple body datasets
"""

from pathlib import Path
import sys

# Import the viewer
from dicom_viewer_3d import DICOM3DViewer

def main():
    # Data is in parent directory (project root)
    project_root = Path(__file__).parent.parent
    
    # Available datasets
    datasets = {
        "1": ("2021.003/DICOM", "Body 1 - Original (2021.003)"),
        "2": ("2021.013/DICOM", "Body 2 (2021.013)"),
        "3": ("2021.014/DICOM", "Body 3 (2021.014)"),
        "4": ("2021.020/DICOM", "Body 4 (2021.020)"),
        "5": ("2021.021/DICOM", "Body 5 (2021.021)"),
        "6": ("2021.022/DICOM", "Body 6 (2021.022)"),
    }
    
    print("="*60)
    print("DICOM 3D CT Scan Viewer")
    print("="*60)
    print("\nAvailable datasets:")
    for key, (path, desc) in datasets.items():
        full_path = project_root / path
        exists = "✓" if full_path.exists() else "✗"
        print(f"  {key}. {desc} [{exists}]")
    
    print("\nEnter dataset number (or 'q' to quit):")
    choice = input("> ").strip()
    
    if choice.lower() == 'q':
        print("Goodbye!")
        sys.exit(0)
    
    if choice not in datasets:
        print(f"Invalid choice: {choice}")
        sys.exit(1)
    
    dicom_path, desc = datasets[choice]
    dicom_dir = project_root / dicom_path
    
    if not dicom_dir.exists():
        print(f"Error: DICOM directory not found at {dicom_dir}")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"Loading: {desc}")
    print(f"Path: {dicom_dir}")
    print("="*60)
    
    # For Body 1, check if segmentations exist and load them
    aorta_seg_path = None
    bone_seg_path = None
    if choice == "1":
        # Check for aorta segmentation
        aorta_path = project_root / "data/segmentations/fullbody_aorta_segmentation.npy"
        if aorta_path.exists():
            aorta_seg_path = aorta_path
            print(f"Found aorta segmentation: {aorta_path.name}")
            print("Will display aorta in RED overlay")
        else:
            print("Note: Aorta segmentation not found")
        
        # Check for bone/organ segmentation
        bone_path = project_root / "data/segmentations/upper_body_bones_segmentation.npy"
        if bone_path.exists():
            bone_seg_path = bone_path
            print(f"Found bone/organ segmentation: {bone_path.name}")
            print("Will display labeled bones with hover tooltips")
        else:
            print("Note: Bone segmentation not found")
    
    # Create viewer (with segmentations for Body 1 if available)
    viewer = DICOM3DViewer(dicom_dir, aorta_seg_path=aorta_seg_path, bone_seg_path=bone_seg_path)
    
    # Show VTK volume rendering
    # For the original body with separate leg scans, use ICP registration
    # For single-series datasets, this will just show the normal volume
    print("\n" + "="*60)
    print("3D Volume Rendering (VTK)")
    print("="*60)
    viewer.show_vtk_volume_icp_registered()
    
    # Ask if user wants more visualizations
    print("\n" + "="*60)
    print("Additional visualizations available:")
    print("  1. Orthogonal Views (Axial/Sagittal/Coronal sliders)")
    print("  2. Maximum Intensity Projection (MIP)")
    print("  3. 3D Surface Reconstruction (browser)")
    print("  q. Quit")
    print("="*60)
    
    while True:
        choice = input("\nEnter choice (1/2/3/q): ").strip().lower()
        
        if choice == 'q':
            break
        elif choice == '1':
            print("Showing Orthogonal Views...")
            viewer.show_orthogonal_views()
        elif choice == '2':
            print("Showing Maximum Intensity Projection...")
            viewer.show_3d_volume_mip()
        elif choice == '3':
            print("Creating 3D Surface Reconstruction...")
            threshold = 300
            print(f"Using threshold: {threshold} HU")
            viewer.show_3d_surface(threshold)
        else:
            print("Invalid choice")
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)

if __name__ == "__main__":
    main()

