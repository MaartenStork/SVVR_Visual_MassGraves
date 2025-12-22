#!/usr/bin/env python3
"""
Complete DICOM Data Viewer
Shows ALL data including scout images, all reconstructions, all series
Nothing is hidden - see everything that's in the DICOM folder
"""

import pydicom
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import sys


class CompleteDICOMViewer:
    def __init__(self, dicom_dir):
        self.dicom_dir = Path(dicom_dir)
        self.datasets_by_size = defaultdict(list)
        self.datasets_by_series = defaultdict(list)
        
        print("Loading ALL DICOM files...")
        self.load_all_files()
        print(f"\nTotal files loaded: {sum(len(v) for v in self.datasets_by_size.values())}")
        
    def load_all_files(self):
        """Load ALL DICOM files, grouped by image size and series"""
        for file_path in self.dicom_dir.glob('*'):
            if file_path.is_file():
                try:
                    ds = pydicom.dcmread(str(file_path))
                    if hasattr(ds, 'pixel_array'):
                        shape = ds.pixel_array.shape
                        series_num = ds.SeriesNumber if hasattr(ds, 'SeriesNumber') else 0
                        
                        # Store by size
                        self.datasets_by_size[shape].append({
                            'dataset': ds,
                            'path': file_path,
                            'series': series_num
                        })
                        
                        # Store by series
                        self.datasets_by_series[series_num].append({
                            'dataset': ds,
                            'path': file_path,
                            'shape': shape
                        })
                except Exception:
                    pass
        
        # Sort each group by instance number
        for shape in self.datasets_by_size:
            self.datasets_by_size[shape].sort(
                key=lambda x: x['dataset'].InstanceNumber 
                if hasattr(x['dataset'], 'InstanceNumber') else 0
            )
        
        for series in self.datasets_by_series:
            self.datasets_by_series[series].sort(
                key=lambda x: x['dataset'].InstanceNumber 
                if hasattr(x['dataset'], 'InstanceNumber') else 0
            )
    
    def show_overview(self):
        """Show overview of all data"""
        print("\n" + "="*80)
        print("COMPLETE DATA OVERVIEW")
        print("="*80)
        
        print("\n--- BY IMAGE SIZE ---")
        for idx, (shape, items) in enumerate(sorted(self.datasets_by_size.items(), 
                                                     key=lambda x: -len(x[1])), 1):
            print(f"\n{idx}. Image size: {shape} - {len(items)} files")
            
            # Group by series within this size
            series_in_size = defaultdict(list)
            for item in items:
                series_in_size[item['series']].append(item)
            
            for series_num in sorted(series_in_size.keys()):
                series_items = series_in_size[series_num]
                ds = series_items[0]['dataset']
                series_desc = ds.SeriesDescription if hasattr(ds, 'SeriesDescription') else 'N/A'
                print(f"   Series {series_num}: {len(series_items)} files - {series_desc}")
        
        print("\n" + "="*80)
        print("\n--- BY SERIES NUMBER ---")
        for series_num in sorted(self.datasets_by_series.keys()):
            items = self.datasets_by_series[series_num]
            ds = items[0]['dataset']
            series_desc = ds.SeriesDescription if hasattr(ds, 'SeriesDescription') else 'N/A'
            shape = items[0]['shape']
            print(f"\nSeries {series_num}: {len(items)} files - {series_desc}")
            print(f"   Image size: {shape}")
        
        print("\n" + "="*80)
    
    def browse_by_size(self):
        """Browse data grouped by image size"""
        sizes = sorted(self.datasets_by_size.keys(), key=lambda x: -len(self.datasets_by_size[x]))
        
        print("\n" + "="*80)
        print("BROWSE BY IMAGE SIZE")
        print("="*80)
        for idx, shape in enumerate(sizes, 1):
            count = len(self.datasets_by_size[shape])
            print(f"{idx}. {shape} ({count} files)")
        
        choice = input(f"\nSelect image size to view (1-{len(sizes)}, or 'q' to quit): ").strip()
        
        if choice.lower() == 'q':
            return
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(sizes):
                selected_shape = sizes[idx]
                self._view_size_group(selected_shape)
            else:
                print("Invalid choice!")
        except ValueError:
            print("Invalid input!")
    
    def browse_by_series(self):
        """Browse data grouped by series number"""
        series_nums = sorted(self.datasets_by_series.keys())
        
        print("\n" + "="*80)
        print("BROWSE BY SERIES NUMBER")
        print("="*80)
        for series_num in series_nums:
            items = self.datasets_by_series[series_num]
            ds = items[0]['dataset']
            series_desc = ds.SeriesDescription if hasattr(ds, 'SeriesDescription') else 'N/A'
            print(f"Series {series_num}: {len(items)} files - {series_desc}")
        
        choice = input(f"\nEnter series number to view (or 'q' to quit): ").strip()
        
        if choice.lower() == 'q':
            return
        
        try:
            series_num = int(choice)
            if series_num in self.datasets_by_series:
                self._view_series_group(series_num)
            else:
                print(f"Series {series_num} not found!")
        except ValueError:
            print("Invalid input!")
    
    def _view_size_group(self, shape):
        """View all images of a specific size"""
        items = self.datasets_by_size[shape]
        
        print(f"\nViewing {len(items)} images of size {shape}")
        print("Use arrow keys (left/right) or A/D to navigate, Q to quit")
        
        current_idx = [0]  # Use list to modify in closure
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        def update_display():
            ax.clear()
            item = items[current_idx[0]]
            ds = item['dataset']
            
            # Get pixel array
            pixel_array = ds.pixel_array
            
            # Get window/level if available
            if hasattr(ds, 'WindowCenter') and hasattr(ds, 'WindowWidth'):
                wc = ds.WindowCenter
                ww = ds.WindowWidth
                if isinstance(wc, (list, pydicom.multival.MultiValue)):
                    wc = float(wc[0])
                else:
                    wc = float(wc)
                if isinstance(ww, (list, pydicom.multival.MultiValue)):
                    ww = float(ww[0])
                else:
                    ww = float(ww)
                vmin = wc - ww / 2
                vmax = wc + ww / 2
            else:
                vmin, vmax = -1000, 3000
            
            # Display
            ax.imshow(pixel_array, cmap='gray', vmin=vmin, vmax=vmax)
            
            # Title with info
            series_num = item['series']
            series_desc = ds.SeriesDescription if hasattr(ds, 'SeriesDescription') else 'N/A'
            instance = ds.InstanceNumber if hasattr(ds, 'InstanceNumber') else 'N/A'
            
            title = f"Image {current_idx[0] + 1}/{len(items)} | Size: {shape}\n"
            title += f"Series {series_num}: {series_desc} | Instance: {instance}"
            
            ax.set_title(title, fontsize=10)
            ax.axis('off')
            fig.canvas.draw()
        
        def on_key(event):
            if event.key in ['right', 'd']:
                current_idx[0] = min(current_idx[0] + 1, len(items) - 1)
                update_display()
            elif event.key in ['left', 'a']:
                current_idx[0] = max(current_idx[0] - 1, 0)
                update_display()
            elif event.key in ['q', 'escape']:
                plt.close()
        
        fig.canvas.mpl_connect('key_press_event', on_key)
        update_display()
        plt.show()
    
    def _view_series_group(self, series_num):
        """View all images from a specific series"""
        items = self.datasets_by_series[series_num]
        ds = items[0]['dataset']
        series_desc = ds.SeriesDescription if hasattr(ds, 'SeriesDescription') else 'N/A'
        shape = items[0]['shape']
        
        print(f"\nViewing Series {series_num}: {series_desc}")
        print(f"Image size: {shape}, {len(items)} files")
        print("Use arrow keys (left/right) or A/D to navigate, Q to quit")
        
        current_idx = [0]
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        def update_display():
            ax.clear()
            item = items[current_idx[0]]
            ds = item['dataset']
            
            # Get pixel array
            pixel_array = ds.pixel_array
            
            # Get window/level if available
            if hasattr(ds, 'WindowCenter') and hasattr(ds, 'WindowWidth'):
                wc = ds.WindowCenter
                ww = ds.WindowWidth
                if isinstance(wc, (list, pydicom.multival.MultiValue)):
                    wc = float(wc[0])
                else:
                    wc = float(wc)
                if isinstance(ww, (list, pydicom.multival.MultiValue)):
                    ww = float(ww[0])
                else:
                    ww = float(ww)
                vmin = wc - ww / 2
                vmax = wc + ww / 2
            else:
                vmin, vmax = -1000, 3000
            
            # Display
            ax.imshow(pixel_array, cmap='gray', vmin=vmin, vmax=vmax)
            
            # Title with info
            instance = ds.InstanceNumber if hasattr(ds, 'InstanceNumber') else 'N/A'
            slice_loc = f"{ds.SliceLocation:.1f}mm" if hasattr(ds, 'SliceLocation') else 'N/A'
            
            title = f"Series {series_num}: {series_desc}\n"
            title += f"Image {current_idx[0] + 1}/{len(items)} | Instance: {instance} | Location: {slice_loc}"
            
            ax.set_title(title, fontsize=10)
            ax.axis('off')
            fig.canvas.draw()
        
        def on_key(event):
            if event.key in ['right', 'd']:
                current_idx[0] = min(current_idx[0] + 1, len(items) - 1)
                update_display()
            elif event.key in ['left', 'a']:
                current_idx[0] = max(current_idx[0] - 1, 0)
                update_display()
            elif event.key in ['q', 'escape']:
                plt.close()
        
        fig.canvas.mpl_connect('key_press_event', on_key)
        update_display()
        plt.show()
    
    def view_scouts_only(self):
        """View only the scout/SPO images"""
        print("\n" + "="*80)
        print("SCOUT/SPO IMAGES VIEWER")
        print("="*80)
        
        # Find all non-512x512 images or scout series
        scout_series = []
        for series_num, items in self.datasets_by_series.items():
            ds = items[0]['dataset']
            series_desc = ds.SeriesDescription if hasattr(ds, 'SeriesDescription') else ''
            shape = items[0]['shape']
            
            # Scout images are usually non-square or have "SPO" in description
            if 'SPO' in series_desc or shape[0] != shape[1]:
                scout_series.append((series_num, items, series_desc, shape))
        
        if not scout_series:
            print("No scout images found!")
            return
        
        print(f"\nFound {len(scout_series)} scout/SPO series:")
        for idx, (series_num, items, desc, shape) in enumerate(scout_series, 1):
            print(f"{idx}. Series {series_num}: {desc} ({shape}, {len(items)} files)")
        
        choice = input(f"\nSelect scout series to view (1-{len(scout_series)}, or 'q' to quit): ").strip()
        
        if choice.lower() == 'q':
            return
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(scout_series):
                series_num = scout_series[idx][0]
                self._view_series_group(series_num)
            else:
                print("Invalid choice!")
        except ValueError:
            print("Invalid input!")


def main():
    # Get the DICOM directory
    script_dir = Path(__file__).parent
    dicom_dir = script_dir / "DICOM"
    
    if not dicom_dir.exists():
        print(f"Error: DICOM directory not found at {dicom_dir}")
        sys.exit(1)
    
    # Create viewer
    viewer = CompleteDICOMViewer(dicom_dir)
    
    # Show overview
    viewer.show_overview()
    
    # Main menu
    while True:
        print("\n" + "="*80)
        print("COMPLETE DICOM DATA VIEWER - MAIN MENU")
        print("="*80)
        print("1. Browse by Image Size")
        print("2. Browse by Series Number")
        print("3. View Scout/SPO Images Only")
        print("4. Show Overview Again")
        print("5. Quit")
        print("="*80)
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            viewer.browse_by_size()
        elif choice == '2':
            viewer.browse_by_series()
        elif choice == '3':
            viewer.view_scouts_only()
        elif choice == '4':
            viewer.show_overview()
        elif choice == '5':
            print("\nExiting...")
            break
        else:
            print("Invalid choice!")


if __name__ == "__main__":
    main()

