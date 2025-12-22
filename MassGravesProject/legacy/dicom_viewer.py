#!/usr/bin/env python3
"""
Simple DICOM CT Scan Viewer
Navigate through CT slices using left/right arrow keys or A/D keys
"""

import os
import sys
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class DICOMViewer:
    def __init__(self, dicom_dir):
        self.dicom_dir = Path(dicom_dir)
        self.current_index = 0
        self.dicom_files = []
        self.datasets = []
        
        print("Loading DICOM files...")
        self.load_dicom_files()
        print(f"Loaded {len(self.dicom_files)} DICOM files")
        
        if not self.dicom_files:
            print("No DICOM files found!")
            sys.exit(1)
            
        # Sort by Instance Number if available
        self.sort_files()
        
        # Setup the plot
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        self.display_image()
        
    def load_dicom_files(self):
        """Load all DICOM files from the directory"""
        for file_path in self.dicom_dir.glob('*'):
            if file_path.is_file():
                try:
                    ds = pydicom.dcmread(str(file_path))
                    # Check if it has pixel data
                    if hasattr(ds, 'pixel_array'):
                        self.dicom_files.append(file_path)
                        self.datasets.append(ds)
                except Exception as e:
                    # Skip files that aren't valid DICOM or can't be read
                    pass
    
    def sort_files(self):
        """Sort files by Instance Number or Slice Location"""
        try:
            # Try to sort by Instance Number
            sorted_indices = sorted(range(len(self.datasets)), 
                                  key=lambda i: int(self.datasets[i].InstanceNumber))
            self.dicom_files = [self.dicom_files[i] for i in sorted_indices]
            self.datasets = [self.datasets[i] for i in sorted_indices]
            print("Sorted by Instance Number")
        except:
            try:
                # Try to sort by Slice Location
                sorted_indices = sorted(range(len(self.datasets)), 
                                      key=lambda i: float(self.datasets[i].SliceLocation))
                self.dicom_files = [self.dicom_files[i] for i in sorted_indices]
                self.datasets = [self.datasets[i] for i in sorted_indices]
                print("Sorted by Slice Location")
            except:
                print("Could not sort files - displaying in directory order")
    
    def display_image(self):
        """Display the current DICOM image"""
        self.ax.clear()
        
        ds = self.datasets[self.current_index]
        
        # Get pixel array
        pixel_array = ds.pixel_array
        
        # Apply window/level for CT images
        if hasattr(ds, 'WindowCenter') and hasattr(ds, 'WindowWidth'):
            # Handle multiple window values (take first)
            window_center = ds.WindowCenter
            window_width = ds.WindowWidth
            if isinstance(window_center, (list, pydicom.multival.MultiValue)):
                window_center = float(window_center[0])
            else:
                window_center = float(window_center)
            if isinstance(window_width, (list, pydicom.multival.MultiValue)):
                window_width = float(window_width[0])
            else:
                window_width = float(window_width)
                
            vmin = window_center - window_width / 2
            vmax = window_center + window_width / 2
        else:
            # Default window for CT
            vmin = -1000  # Air
            vmax = 3000   # Bone
        
        # Display the image
        self.ax.imshow(pixel_array, cmap='gray', vmin=vmin, vmax=vmax)
        
        # Add image information
        title = f"Slice {self.current_index + 1}/{len(self.dicom_files)}"
        
        # Add patient and study info
        info_lines = []
        if hasattr(ds, 'PatientName'):
            info_lines.append(f"Patient: {ds.PatientName}")
        if hasattr(ds, 'StudyDescription'):
            info_lines.append(f"Study: {ds.StudyDescription}")
        if hasattr(ds, 'SeriesDescription'):
            info_lines.append(f"Series: {ds.SeriesDescription}")
        if hasattr(ds, 'SliceLocation'):
            info_lines.append(f"Location: {ds.SliceLocation:.2f} mm")
        if hasattr(ds, 'SliceThickness'):
            info_lines.append(f"Thickness: {ds.SliceThickness} mm")
        
        title += "\n" + " | ".join(info_lines)
        
        self.ax.set_title(title, fontsize=10, pad=10)
        self.ax.axis('off')
        
        plt.draw()
    
    def on_key(self, event):
        """Handle keyboard events"""
        if event.key == 'right' or event.key == 'd':
            self.next_image()
        elif event.key == 'left' or event.key == 'a':
            self.previous_image()
        elif event.key == 'q' or event.key == 'escape':
            plt.close()
    
    def next_image(self):
        """Go to next image"""
        if self.current_index < len(self.dicom_files) - 1:
            self.current_index += 1
            self.display_image()
    
    def previous_image(self):
        """Go to previous image"""
        if self.current_index > 0:
            self.current_index -= 1
            self.display_image()
    
    def show(self):
        """Show the viewer"""
        print("\nControls:")
        print("  Left Arrow / A : Previous slice")
        print("  Right Arrow / D : Next slice")
        print("  Q / ESC : Quit")
        plt.show()


def main():
    # Get the DICOM directory
    script_dir = Path(__file__).parent
    dicom_dir = script_dir / "DICOM"
    
    if not dicom_dir.exists():
        print(f"Error: DICOM directory not found at {dicom_dir}")
        sys.exit(1)
    
    # Create and show viewer
    viewer = DICOMViewer(dicom_dir)
    viewer.show()


if __name__ == "__main__":
    main()

