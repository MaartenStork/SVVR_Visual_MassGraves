#!/usr/bin/env python3
"""
3D DICOM CT Scan Viewer with Volume Reconstruction
Supports multiple viewing modes: 2D slices, 3D volume, and orthogonal views
"""

import os
import sys
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from pathlib import Path
from scipy import ndimage
from skimage.metrics import structural_similarity
import plotly.graph_objects as go
import plotly.express as px


class DICOM3DViewer:
    def __init__(self, dicom_dir, aorta_seg_path=None, bone_seg_path=None):
        self.dicom_dir = Path(dicom_dir)
        self.dicom_files = []
        self.datasets = []
        self.volume = None
        self.spacing = None
        self.aorta_mask = None
        self.aorta_spacing = None
        self.aorta_origin = None
        # Bone segmentation data
        self.bone_mask = None
        self.bone_spacing = None
        self.bone_origin = None
        self.bone_labels = None  # Dict mapping label_value -> {name, color}
        # Store CT volume's world coordinate reference for alignment
        self.ct_world_origin = None  # DICOM coordinates of VTK origin (0,0,0)
        
        print("Loading DICOM files...")
        self.load_dicom_files()
        print(f"Loaded {len(self.dicom_files)} DICOM files")
        
        # Load aorta segmentation if provided
        if aorta_seg_path:
            self.load_aorta_segmentation(aorta_seg_path)
        
        # Load bone segmentation if provided
        if bone_seg_path:
            self.load_bone_segmentation(bone_seg_path)
        
        if not self.dicom_files:
            print("No DICOM files found!")
            sys.exit(1)
            
        # Sort by Instance Number if available
        self.sort_files()
        
        # Build 3D volume
        print("Building 3D volume...")
        self.build_volume()
        print(f"Volume shape: {self.volume.shape}")
        print(f"Volume spacing: {self.spacing}")
        
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
                except Exception:
                    pass
    
    def load_aorta_segmentation(self, seg_path):
        """Load aorta segmentation from .npy file with corresponding header"""
        import json
        
        seg_path = Path(seg_path)
        print(f"\nLoading aorta segmentation from {seg_path}...")
        
        # Load the segmentation mask
        self.aorta_mask = np.load(seg_path)
        print(f"Aorta mask shape: {self.aorta_mask.shape}")
        print(f"Aorta voxels: {np.count_nonzero(self.aorta_mask)}")
        
        # Load the corresponding header - try different naming patterns
        header_path = None
        # Try pattern: *_segmentation.npy -> *_header.json (fullbody style)
        if '_segmentation.npy' in seg_path.name:
            header_path = seg_path.parent / seg_path.name.replace('_segmentation.npy', '_header.json')
        # Try pattern: *_segmentation_N.npy -> *_header_N.json (old style)
        if header_path is None or not header_path.exists():
            header_path = seg_path.parent / seg_path.name.replace('_segmentation_', '_header_').replace('.npy', '.json')
        
        if header_path.exists():
            with open(header_path, 'r') as f:
                header = json.load(f)
            
            # Extract spacing from space directions matrix (diagonal values)
            space_dirs = header['space directions']
            self.aorta_spacing = [
                abs(space_dirs[0][0]),  # X spacing
                abs(space_dirs[1][1]),  # Y spacing
                abs(space_dirs[2][2])   # Z spacing
            ]
            
            # Extract origin
            self.aorta_origin = header['space origin']
            
            print(f"Aorta spacing: {self.aorta_spacing} mm")
            print(f"Aorta origin: {self.aorta_origin} mm")
        else:
            print(f"Warning: Header file not found at {header_path}")
            self.aorta_spacing = [1.0, 1.0, 1.0]
            self.aorta_origin = [0.0, 0.0, 0.0]
    
    def load_bone_segmentation(self, seg_path):
        """Load bone segmentation from .npy file with corresponding header and labels"""
        import json
        
        seg_path = Path(seg_path)
        print(f"\nLoading bone segmentation from {seg_path}...")
        
        # Load the segmentation mask (labeled - each bone has unique int value)
        self.bone_mask = np.load(seg_path)
        print(f"Bone mask shape: {self.bone_mask.shape}")
        unique_labels = np.unique(self.bone_mask)
        print(f"Unique labels: {len(unique_labels)} (including background)")
        
        # Load the corresponding header
        header_path = seg_path.parent / seg_path.name.replace('_segmentation.npy', '_header.json')
        labels_path = seg_path.parent / seg_path.name.replace('_segmentation.npy', '_labels.json')
        
        if header_path.exists():
            with open(header_path, 'r') as f:
                header = json.load(f)
            
            space_dirs = header['space directions']
            self.bone_spacing = [
                abs(space_dirs[0][0]),
                abs(space_dirs[1][1]),
                abs(space_dirs[2][2])
            ]
            self.bone_origin = header['space origin']
            
            print(f"Bone spacing: {self.bone_spacing} mm")
            print(f"Bone origin: {self.bone_origin} mm")
        else:
            print(f"Warning: Header file not found at {header_path}")
            self.bone_spacing = [1.0, 1.0, 1.0]
            self.bone_origin = [0.0, 0.0, 0.0]
        
        # Load labels
        if labels_path.exists():
            with open(labels_path, 'r') as f:
                self.bone_labels = json.load(f)
            print(f"Loaded {len(self.bone_labels)} segment labels")
        else:
            print(f"Warning: Labels file not found at {labels_path}")
            self.bone_labels = {}
    
    def sort_files(self):
        """Sort files by ImagePositionPatient Z-coordinate (for proper head-to-toe ordering)"""
        try:
            # First, try to sort by ImagePositionPatient Z-coordinate
            # This handles cases where scans are done in multiple sessions
            sorted_indices = sorted(range(len(self.datasets)), 
                                  key=lambda i: float(self.datasets[i].ImagePositionPatient[2]) 
                                  if hasattr(self.datasets[i], 'ImagePositionPatient') else float('inf'),
                                  reverse=True)  # Reverse=True for head-to-toe (high Z to low Z)
            self.dicom_files = [self.dicom_files[i] for i in sorted_indices]
            self.datasets = [self.datasets[i] for i in sorted_indices]
            print("Sorted by ImagePositionPatient (Z-coordinate) for head-to-toe ordering")
        except:
            try:
                sorted_indices = sorted(range(len(self.datasets)), 
                                      key=lambda i: float(self.datasets[i].SliceLocation),
                                      reverse=True)  # Head to toes
                self.dicom_files = [self.dicom_files[i] for i in sorted_indices]
                self.datasets = [self.datasets[i] for i in sorted_indices]
                print("Sorted by Slice Location (head-to-toe)")
            except:
                try:
                    sorted_indices = sorted(range(len(self.datasets)), 
                                          key=lambda i: int(self.datasets[i].InstanceNumber))
                    self.dicom_files = [self.dicom_files[i] for i in sorted_indices]
                    self.datasets = [self.datasets[i] for i in sorted_indices]
                    print("Sorted by Instance Number")
                except:
                    print("Could not sort files - using directory order")
    
    def build_volume(self):
        """Build a 3D volume from all slices"""
        # Group slices by shape to handle scout/localizer images
        from collections import Counter, defaultdict
        
        shapes = [ds.pixel_array.shape for ds in self.datasets]
        shape_counts = Counter(shapes)
        
        # Find the most common shape (this is likely the actual CT series)
        most_common_shape = shape_counts.most_common(1)[0][0]
        
        print(f"Found {len(shape_counts)} different image sizes")
        print(f"Most common shape: {most_common_shape} ({shape_counts[most_common_shape]} slices)")
        
        # Filter datasets to only include those with the most common shape
        filtered_datasets = []
        filtered_files = []
        for ds, file_path in zip(self.datasets, self.dicom_files):
            if ds.pixel_array.shape == most_common_shape:
                filtered_datasets.append(ds)
                filtered_files.append(file_path)
        
        # Group by SeriesNumber to separate multiple scan sessions
        series_groups = defaultdict(list)
        for ds, file_path in zip(filtered_datasets, filtered_files):
            series_num = ds.SeriesNumber if hasattr(ds, 'SeriesNumber') else 0
            series_groups[series_num].append((ds, file_path))
        
        print(f"\nFound {len(series_groups)} different series:")
        series_info = []
        for series_num, items in series_groups.items():
            datasets_in_series = [item[0] for item in items]
            z_positions = [float(ds.ImagePositionPatient[2]) for ds in datasets_in_series 
                          if hasattr(ds, 'ImagePositionPatient')]
            if z_positions and len(items) > 5:  # Only consider series with more than 5 slices
                series_desc = datasets_in_series[0].SeriesDescription if hasattr(datasets_in_series[0], 'SeriesDescription') else 'N/A'
                print(f"  Series {series_num}: {len(items)} slices, Z range: {min(z_positions):.1f} to {max(z_positions):.1f} - {series_desc}")
                series_info.append((series_num, min(z_positions), max(z_positions), len(items), items))
        
        # Check if we have multiple non-overlapping series that should be concatenated
        # This handles the case where the person was scanned in two separate sessions
        if len(series_info) >= 2:
            # Group series by similar Z ranges (to identify duplicates/reconstructions of same scan)
            def ranges_overlap(range1, range2, tolerance=50):
                """Check if two Z-ranges overlap significantly"""
                min1, max1 = range1
                min2, max2 = range2
                overlap = min(max1, max2) - max(min1, min2)
                range1_size = max1 - min1
                range2_size = max2 - min2
                # Overlapping if more than 80% of the smaller range overlaps
                return overlap > 0.8 * min(range1_size, range2_size)
            
            # Group overlapping series together
            series_groups_by_range = []
            for series_num, min_z, max_z, count, items in series_info:
                found_group = False
                for group in series_groups_by_range:
                    if ranges_overlap((min_z, max_z), (group[0][1], group[0][2])):
                        group.append((series_num, min_z, max_z, count, items))
                        found_group = True
                        break
                if not found_group:
                    series_groups_by_range.append([(series_num, min_z, max_z, count, items)])
            
            print(f"\nGrouped into {len(series_groups_by_range)} distinct anatomical regions:")
            
            # Select the best series from each group (usually the one with most slices)
            selected_series = []
            for i, group in enumerate(series_groups_by_range):
                # Sort by number of slices (descending)
                group.sort(key=lambda x: x[3], reverse=True)
                best = group[0]
                print(f"  Region {i+1}: Using series {best[0]} ({best[3]} slices, Z: {best[1]:.1f} to {best[2]:.1f})")
                selected_series.append(best)
            
            # Sort selected series by Z position (head to toe)
            selected_series.sort(key=lambda x: x[1], reverse=False)  # Sort by min Z (legs first, then head)
            
            # Check for gaps between selected series
            if len(selected_series) >= 2:
                gaps = []
                for i in range(len(selected_series) - 1):
                    current_min = selected_series[i][1]  # min Z of current series
                    next_max = selected_series[i+1][2]   # max Z of next series
                    gap = current_min - next_max
                    gaps.append(gap)
                
                print(f"\nGaps between selected regions: {[f'{g:.1f}mm' for g in gaps]}")
                
                # Calculate total anatomical coverage if concatenated
                total_range_concatenated = selected_series[0][2] - selected_series[-1][1]
                largest_single_range = max([s[2] - s[1] for s in selected_series])
                
                print(f"Total anatomical range if concatenated: {total_range_concatenated:.1f}mm")
                print(f"Largest single series range: {largest_single_range:.1f}mm")
                
                # If concatenating gives us significantly more coverage (>20% more), 
                # then these are likely separate scan sessions that should be combined
                if total_range_concatenated > largest_single_range * 1.2:
                    print("Detected separate scan sessions covering different anatomical regions")
                    print("→ Concatenating in head-to-toe order")
                    # Flag that we need to find optimal cut point
                    self._needs_overlap_removal = True
                elif any(abs(gap) > 50 for gap in gaps):
                    print("Detected significant gap between series - concatenating")
                    self._needs_overlap_removal = True
                else:
                    print("Series appear to be reconstructions of same scan - using largest")
                    selected_series = [selected_series[0]]
                    self._needs_overlap_removal = False
            
            # Check for different pixel spacings between series and resample if needed
            print("\nChecking pixel spacing consistency...")
            series_pixel_spacings = []
            series_positions = []
            for series_num, min_z, max_z, count, items in selected_series:
                ps = items[0][0].PixelSpacing if hasattr(items[0][0], 'PixelSpacing') else [1.0, 1.0]
                ipp = items[0][0].ImagePositionPatient if hasattr(items[0][0], 'ImagePositionPatient') else [0, 0, 0]
                series_pixel_spacings.append((series_num, float(ps[0]), float(ps[1])))
                series_positions.append((series_num, float(ipp[0]), float(ipp[1])))
                print(f"  Series {series_num}: {float(ps[0]):.4f} mm/pixel, Position: ({float(ipp[0]):.1f}, {float(ipp[1]):.1f})")
            
            # Find the smallest pixel spacing (highest resolution)
            target_spacing = min(series_pixel_spacings, key=lambda x: x[1])
            print(f"\nTarget pixel spacing: {target_spacing[1]:.4f} mm/pixel (Series {target_spacing[0]})")
            
            # Calculate reference position (use target series position as reference)
            reference_position = next(pos for pos in series_positions if pos[0] == target_spacing[0])
            print(f"Reference position: Series {reference_position[0]} at ({reference_position[1]:.1f}, {reference_position[2]:.1f})")
            
            # Concatenate and resample if needed
            self.datasets = []
            self.dicom_files = []
            
            for series_num, min_z, max_z, count, items in selected_series:
                # Sort items within each series by Z position (head to toe - high Z to low Z)
                items_sorted = sorted(items, key=lambda x: float(x[0].ImagePositionPatient[2]) 
                                    if hasattr(x[0], 'ImagePositionPatient') else 0, reverse=True)
                
                # Check if this series needs resampling
                current_ps = float(items_sorted[0][0].PixelSpacing[0]) if hasattr(items_sorted[0][0], 'PixelSpacing') else 1.0
                current_pos = next(pos for pos in series_positions if pos[0] == series_num)
                
                # Calculate offset in pixels (for alignment)
                offset_x_mm = current_pos[1] - reference_position[1]
                offset_y_mm = current_pos[2] - reference_position[2]
                offset_x_px = int(round(offset_x_mm / target_spacing[1]))
                offset_y_px = int(round(offset_y_mm / target_spacing[1]))
                
                if abs(current_ps - target_spacing[1]) > 0.001:  # Different spacing
                    scale_factor = current_ps / target_spacing[1]
                    print(f"\n⚠️  Series {series_num} needs rescaling: {current_ps:.4f} → {target_spacing[1]:.4f} mm/pixel")
                    print(f"   Scale factor: {scale_factor:.4f}")
                    print(f"   Position offset: ({offset_x_px}, {offset_y_px}) pixels")
                    
                    # Store datasets with resampling flag and offset
                    for ds, file_path in items_sorted:
                        ds._needs_resampling = True
                        ds._scale_factor = scale_factor
                        ds._target_spacing = target_spacing[1]
                        ds._offset_x = offset_x_px
                        ds._offset_y = offset_y_px
                        self.datasets.append(ds)
                        self.dicom_files.append(file_path)
                else:
                    # No resampling needed but still apply offset
                    print(f"\n  Series {series_num}: No rescaling needed")
                    if offset_x_px != 0 or offset_y_px != 0:
                        print(f"   Position offset: ({offset_x_px}, {offset_y_px}) pixels")
                    for ds, file_path in items_sorted:
                        ds._needs_resampling = False
                        ds._offset_x = offset_x_px
                        ds._offset_y = offset_y_px
                        self.datasets.append(ds)
                        self.dicom_files.append(file_path)
        else:
            # Single series - use as is
            self.datasets = filtered_datasets
            self.dicom_files = filtered_files
            self._needs_overlap_removal = False
        
        print(f"\nUsing {len(self.datasets)} slices for 3D reconstruction")
        
        # Stack all slices and apply rescale if needed
        slices = []
        resampled_count = 0
        max_shape = [0, 0]
        
        # First pass: resample and find max dimensions + offsets
        temp_slices = []
        min_offset_x = 0
        min_offset_y = 0
        max_offset_x = 0
        max_offset_y = 0
        
        for idx, ds in enumerate(self.datasets):
            pixel_array = ds.pixel_array.astype(np.float32)
            
            # Apply rescale intercept and slope to convert to Hounsfield Units
            if hasattr(ds, 'RescaleIntercept') and hasattr(ds, 'RescaleSlope'):
                pixel_array = pixel_array * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
            
            # Resample if needed
            if hasattr(ds, '_needs_resampling') and ds._needs_resampling:
                from scipy import ndimage
                # Zoom to match target resolution
                pixel_array = ndimage.zoom(pixel_array, ds._scale_factor, order=1)
                resampled_count += 1
                
                if idx == 0 or (idx > 0 and not hasattr(self.datasets[idx-1], '_needs_resampling')):
                    # First slice that needs resampling - print info
                    print(f"  Resampling slice {idx+1}: {ds.pixel_array.shape} → {pixel_array.shape}")
            
            # Track offsets for proper canvas size calculation
            offset_x = ds._offset_x if hasattr(ds, '_offset_x') else 0
            offset_y = ds._offset_y if hasattr(ds, '_offset_y') else 0
            
            temp_slices.append((pixel_array, offset_x, offset_y))
            
            # Track min/max offsets
            min_offset_x = min(min_offset_x, offset_x)
            min_offset_y = min(min_offset_y, offset_y)
            max_offset_x = max(max_offset_x, offset_x + pixel_array.shape[1])
            max_offset_y = max(max_offset_y, offset_y + pixel_array.shape[0])
        
        if resampled_count > 0:
            print(f"✓ Resampled {resampled_count} slices to match pixel spacing")
        
        # Calculate canvas size that fits all images with their offsets
        canvas_width = max_offset_x - min_offset_x
        canvas_height = max_offset_y - min_offset_y
        
        print(f"  Offset range: X[{min_offset_x}, {max_offset_x}], Y[{min_offset_y}, {max_offset_y}]")
        print(f"  Canvas size: {canvas_height}×{canvas_width}")
        
        # Second pass: place all slices on canvas with proper alignment
        for pixel_array, offset_x, offset_y in temp_slices:
            # Create canvas filled with air
            padded = np.full((canvas_height, canvas_width), -1000, dtype=np.float32)
            
            # Calculate position on canvas (adjust for negative offsets)
            y_pos = offset_y - min_offset_y
            x_pos = offset_x - min_offset_x
            
            # Place image on canvas
            padded[y_pos:y_pos+pixel_array.shape[0], 
                   x_pos:x_pos+pixel_array.shape[1]] = pixel_array
            
            slices.append(padded)
        
        print(f"✓ Aligned slices using physical position offsets")
        print(f"✓ Final unified shape: {canvas_height}×{canvas_width}")
        
        self.volume = np.stack(slices, axis=0)
        
        # Get spacing information - use the target (smallest/best) pixel spacing
        # Check if we did any resampling - if so, use target spacing
        if hasattr(self.datasets[0], '_target_spacing'):
            target_ps = self.datasets[0]._target_spacing
            pixel_spacing = [target_ps, target_ps]
            print(f"Using target pixel spacing (after resampling): {pixel_spacing} mm")
        else:
            # Use most common pixel spacing
            from collections import Counter
            pixel_spacings = [tuple(ds.PixelSpacing) if hasattr(ds, 'PixelSpacing') else (1.0, 1.0) 
                             for ds in self.datasets[:100]]  # Check first 100
            most_common_spacing = Counter(pixel_spacings).most_common(1)[0][0]
            pixel_spacing = list(most_common_spacing)
            print(f"Using pixel spacing: {pixel_spacing} mm")
        
        # Calculate slice spacing - use SliceThickness first as it's more reliable
        ds = self.datasets[0]  # Get reference to first dataset
        if hasattr(ds, 'SliceThickness'):
            slice_spacing = float(ds.SliceThickness)
            print(f"Using SliceThickness: {slice_spacing} mm")
        elif len(self.datasets) > 10:
            # Calculate median spacing from first 50 consecutive slices for robustness
            try:
                spacings = []
                for i in range(min(50, len(self.datasets) - 1)):
                    if hasattr(self.datasets[i], 'SliceLocation') and hasattr(self.datasets[i+1], 'SliceLocation'):
                        spacing = abs(float(self.datasets[i+1].SliceLocation) - float(self.datasets[i].SliceLocation))
                        if 0.1 < spacing < 50:  # Sanity check: typical CT spacing is 0.5-10mm
                            spacings.append(spacing)
                if spacings:
                    slice_spacing = np.median(spacings)
                    print(f"Calculated median slice spacing: {slice_spacing} mm from {len(spacings)} samples")
                else:
                    slice_spacing = 1.0
            except:
                slice_spacing = 1.0
        else:
            slice_spacing = 1.0
        
        self.spacing = [slice_spacing, float(pixel_spacing[0]), float(pixel_spacing[1])]
        
        # Store the world coordinate origin for alignment with segmentations
        # VTK origin (0,0,0) corresponds to the FIRST slice in the volume at canvas (0,0)
        # 
        # The volume is built by stacking: Series 4 first, then Series 5
        # So VTK Z=0 corresponds to the first slice of Series 4 (NOT the highest Z overall!)
        # 
        # For X,Y: canvas (0,0) is where the min_offset series data starts (Series 4)
        # For Z: the first slice in self.datasets (after processing) is at VTK Z=0
        
        first_ds = self.datasets[0]
        if hasattr(first_ds, 'ImagePositionPatient'):
            first_slice_z = float(first_ds.ImagePositionPatient[2])
        else:
            first_slice_z = 0.0
        
        # Find a dataset that has the minimum offset (this will be at canvas 0,0)
        canvas_origin_x = None
        canvas_origin_y = None
        for ds in self.datasets:
            if hasattr(ds, '_offset_x') and hasattr(ds, '_offset_y'):
                if ds._offset_x == min_offset_x and ds._offset_y == min_offset_y:
                    if hasattr(ds, 'ImagePositionPatient'):
                        canvas_origin_x = float(ds.ImagePositionPatient[0])
                        canvas_origin_y = float(ds.ImagePositionPatient[1])
                        break
        
        # Fallback if we couldn't find the origin
        if canvas_origin_x is None:
            if hasattr(first_ds, 'ImagePositionPatient'):
                canvas_origin_x = float(first_ds.ImagePositionPatient[0])
                canvas_origin_y = float(first_ds.ImagePositionPatient[1])
            else:
                canvas_origin_x = 0.0
                canvas_origin_y = 0.0
            
        self.ct_world_origin = [canvas_origin_x, canvas_origin_y, first_slice_z]
        print(f"CT world origin (DICOM coords at VTK 0,0,0): {self.ct_world_origin}")
        
        # Remove overlap if needed (find optimal cut point between scans)
        if hasattr(self, '_needs_overlap_removal') and self._needs_overlap_removal:
            self.remove_scan_overlap()
        
        if hasattr(self, '_needs_overlap_removal') and self._needs_overlap_removal:
            self.align_sessions_2d_bone()
    
    def segment_bone_volume(self,
                        low_thr=150,     # permissive bone-ish threshold
                        high_thr=300,    # definite bone / calc threshold
                        min_low_size=500):
        """
        Two-threshold bone segmentation with conservative filtering:
        - low_thr: broad bone mask (keeps ribs, fingers, etc.)
        - high_thr: definite bone / calcification
        - min_low_size: minimum size for a low-threshold component to be
            considered a real bone region.

        Returns a boolean mask of bone voxels.
        """
        from scipy import ndimage
        import numpy as np

        vol = self.volume

        # 1. Broad bone-ish mask, slightly closed to connect small gaps
        raw_low = vol > low_thr
        structure = ndimage.generate_binary_structure(3, 2)  # 18-connectivity
        mask_low = ndimage.binary_closing(raw_low, structure=structure, iterations=1)

        # 2. Connected components on low-threshold mask
        labels_low, num_low = ndimage.label(mask_low, structure=structure)
        if num_low == 0:
            print("No low-threshold components found.")
            bone_mask = vol > high_thr
            self._bone_mask = bone_mask.astype(bool)
            return self._bone_mask

        sizes = np.bincount(labels_low.ravel())
        keep_low = sizes >= min_low_size
        keep_low[0] = False  # background

        big_low_components = keep_low[labels_low]
        print(f"Low-threshold components: {num_low}, keeping {keep_low.sum()-1} with size ≥ {min_low_size}")

        # 3. High-threshold mask
        mask_high = vol > high_thr

        # 4. Final skeleton: high-HU voxels inside big low-HU bone regions
        bone_mask = mask_high & big_low_components

        self._bone_mask = bone_mask.astype(bool)
        return self._bone_mask
    
    def remove_scan_overlap(self):
        """Find optimal cut point between overlapping scans using image registration"""
        print("\n" + "="*80)
        print("REMOVING SCAN OVERLAP - FINDING OPTIMAL CUT POINT")
        print("="*80)
        
        # Find the transition point between the two series in our dataset list
        transition_idx = None
        first_series = self.datasets[0].SeriesNumber if hasattr(self.datasets[0], 'SeriesNumber') else None
        
        for i in range(1, len(self.datasets)):
            current_series = self.datasets[i].SeriesNumber if hasattr(self.datasets[i], 'SeriesNumber') else None
            if current_series != first_series:
                transition_idx = i
                break
        
        if transition_idx is None:
            print("No series transition found - skipping overlap removal")
            return
        
        print(f"Series transition at slice {transition_idx}")
        print(f"Upper body: slices 0-{transition_idx-1}")
        print(f"Lower body: slices {transition_idx}-{len(self.datasets)-1}")
        
        # Define search region - limit to reasonable overlap (~400mm max = 80 slices at 5mm)
        # Real anatomical overlap between scans is typically 200-400mm
        # IMPORTANT: Only search in the SECOND series (after transition) to find overlap
        second_series_length = len(self.datasets) - transition_idx
        max_reasonable_overlap = 80  # ~400mm at 5mm slices
        search_window_after = min(max_reasonable_overlap, second_series_length - 10)
        
        # Start search AT the transition point, not before it
        start_idx = transition_idx
        end_idx = min(len(self.datasets), transition_idx + search_window_after)
        
        search_range_mm = (end_idx - start_idx) * self.spacing[0]
        print(f"\nSearching for optimal cut point in range [{start_idx}, {end_idx}]")
        print(f"Searching {end_idx - start_idx} slices ({search_range_mm:.1f}mm = {search_range_mm/10:.1f}cm)")
        
        # Coarse search: test every 5th slice for speed (since we're searching so much)
        print("\n[1/2] Coarse search (every 5 slices)...")
        best_score = -np.inf
        best_cut = transition_idx
        comparison_slices = 10  # Compare even more slices for robust matching
        
        scores = []
        for cut_idx in range(start_idx, end_idx, 5):  # Every 5th slice for speed
            if cut_idx < comparison_slices or cut_idx >= len(self.volume) - comparison_slices:
                continue
            
            # Get slices before and after potential cut
            before_slices = self.volume[cut_idx - comparison_slices:cut_idx]
            after_slices = self.volume[cut_idx:cut_idx + comparison_slices]
            
            # Calculate similarity using normalized cross-correlation + SSIM
            score = self._calculate_overlap_score(before_slices, after_slices)
            scores.append((cut_idx, score))
            
            if score > best_score:
                best_score = score
                best_cut = cut_idx
        
        print(f"   Best coarse match at slice {best_cut} (score: {best_score:.4f})")
        
        # Fine search: zoom in around best coarse match
        # Stay within the second series (don't go below transition_idx)
        print("\n[2/2] Fine search (every slice)...")
        fine_start = max(transition_idx, best_cut - 6)
        fine_end = min(end_idx, best_cut + 6)
        
        best_score = -np.inf
        final_cut = best_cut
        
        for cut_idx in range(fine_start, fine_end):
            if cut_idx < comparison_slices or cut_idx >= len(self.volume) - comparison_slices:
                continue
            
            before_slices = self.volume[cut_idx - comparison_slices:cut_idx]
            after_slices = self.volume[cut_idx:cut_idx + comparison_slices]
            
            score = self._calculate_overlap_score(before_slices, after_slices)
            scores.append((cut_idx, score))
            
            if score > best_score:
                best_score = score
                final_cut = cut_idx
        
        print(f"   Best fine match at slice {final_cut} (score: {best_score:.4f})")
        
        # ALWAYS keep all of upper body (0:transition_idx)
        # Cut away the overlapping top portion of lower body
        # final_cut tells us where in the COMBINED volume the best match is
        # We want to remove lower body slices from transition_idx up to final_cut
        
        original_slices = len(self.volume)
        
        if final_cut >= transition_idx:
            # Cut is in the lower body section
            # Remove overlapping slices from start of lower body (transition_idx to final_cut)
            overlap_slices = final_cut - transition_idx
            
            self.volume = np.concatenate([
                self.volume[:transition_idx],  # ALL of upper body
                self.volume[final_cut:]  # Lower body minus overlap
            ], axis=0)
            
            self.datasets = (self.datasets[:transition_idx] + 
                           self.datasets[final_cut:])
            self.dicom_files = (self.dicom_files[:transition_idx] + 
                              self.dicom_files[final_cut:])
            
            removed_slices = original_slices - len(self.volume)
            removed_mm = removed_slices * self.spacing[0]
            
            print("\n" + "="*80)
            print("✓ OVERLAP REMOVAL COMPLETE")
            print("="*80)
            print(f"Original slices: {original_slices}")
            print(f"Final slices: {len(self.volume)}")
            print(f"Removed: {removed_slices} slices ({removed_mm:.1f}mm) from TOP of lower body")
            print(f"Kept: ALL of upper body (0:{transition_idx}) + Lower body from slice {final_cut} onward")
            print(f"Removed overlapping slices {transition_idx} to {final_cut-1} from lower body")
        else:
            # Cut is before transition (shouldn't happen if we search properly, but handle it)
            print("\n⚠️  Warning: Optimal cut found BEFORE transition point")
            print(f"   This suggests the scans don't overlap as expected")
            print(f"   Keeping all slices")
        
        print("="*80 + "\n")
    
    def align_sessions_2d_bone(self):
        """Align upper and lower scan sessions in both X and Y directions using bone-overlap metric."""
        print("\n=== 2D Bone-Based Alignment Between Sessions ===")

        # Find transition index
        transition_idx = None
        base_series = self.datasets[0].SeriesNumber if hasattr(self.datasets[0], 'SeriesNumber') else None

        for i in range(1, len(self.datasets)):
            if hasattr(self.datasets[i], 'SeriesNumber') and self.datasets[i].SeriesNumber != base_series:
                transition_idx = i
                break

        if transition_idx is None:
            print("Only one scan session detected — no alignment needed.")
            return

        upper = self.volume[:transition_idx]
        lower = self.volume[transition_idx:]

        # Bone masks
        u_slice = (upper[-1] > 200).astype(np.uint8)
        l_slice = (lower[0] > 200).astype(np.uint8)

        max_shift = 80  # you can tune this
        best_score = -1e9
        best_dx = 0
        best_dy = 0

        print(f"Searching dx, dy from -{max_shift} to +{max_shift}...")

        for dy in range(-max_shift, max_shift + 1):
            rolled_y = np.roll(l_slice, dy, axis=0)

            for dx in range(-max_shift, max_shift + 1):
                rolled_xy = np.roll(rolled_y, dx, axis=1)
                score = np.sum(rolled_xy & u_slice)

                if score > best_score:
                    best_score = score
                    best_dx = dx
                    best_dy = dy

        print(f"Best shifts: dx={best_dx}, dy={best_dy}, score={best_score}")

        # Apply shift to full lower block
        shifted_lower = np.roll(lower, shift=(0, best_dy, best_dx), axis=(0, 1, 2))

        # Update volume
        self.volume = np.concatenate([upper, shifted_lower], axis=0)

        print("✓ Applied 2D bone alignment.")
    
    def _calculate_overlap_score(self, slices1, slices2):
        """
        Calculate similarity score between two 3D blocks of slices
        using bone mask intersection-over-union (IoU).

        Higher score = better match.
        """

        try:
            # Threshold to bone (HU > 200)
            bone1 = slices1 > 300
            bone2 = slices2 > 300

            inter = np.logical_and(bone1, bone2).sum()
            union = np.logical_or(bone1, bone2).sum()

            if union == 0:
                # No bone at all -> meaningless comparison
                return -np.inf

            iou = inter / union  # in [0,1], higher is better

            return iou

        except Exception:
            return -np.inf
    
    def _find_series_transition_index(self):
        """
        Find first index where SeriesNumber changes.
        Returns index or None if only one series.
        """
        if not self.datasets:
            return None

        first_series = getattr(self.datasets[0], "SeriesNumber", None)
        for i, ds in enumerate(self.datasets[1:], start=1):
            if getattr(ds, "SeriesNumber", None) != first_series:
                return i

        return None
    def _vtk_image_from_block(self, block, z0_mm=0.0):
        """
        Convert a 3D NumPy block (z, y, x) to vtkImageData using self.spacing.
        z0_mm is the physical z-position (in mm) of slice index 0 of this block.
        """
        import vtk
        from vtk.util import numpy_support as nps
        import numpy as np

        vol = block.astype(np.int16, copy=False)
        nz, ny, nx = vol.shape

        flat = vol.ravel(order="C")
        vtk_array = nps.numpy_to_vtk(flat, deep=True, array_type=vtk.VTK_SHORT)

        img = vtk.vtkImageData()
        img.SetDimensions(nx, ny, nz)
        img.SetSpacing(self.spacing[2], self.spacing[1], self.spacing[0])
        img.SetOrigin(0.0, 0.0, z0_mm)   # <<–– important
        img.GetPointData().SetScalars(vtk_array)

        return img 
    
    def _compute_icp_transform(self, bone_iso=300, overlap_slices=80):
        import vtk
        import numpy as np

        transition = self._find_series_transition_index()
        if transition is None:
            print("Only one series – no ICP alignment needed.")
            tf = vtk.vtkTransform()
            tf.Identity()
            return tf

        upper = self.volume[:transition]
        lower = self.volume[transition:]

        if upper.shape[0] < 10 or lower.shape[0] < 10:
            print("Not enough slices on one side for ICP.")
            tf = vtk.vtkTransform()
            tf.Identity()
            return tf

        ov = min(overlap_slices, upper.shape[0], lower.shape[0])

        # overlap blocks in NumPy
        upper_ov = upper[-ov:]      # slices [transition-ov .. transition-1]
        lower_ov = lower[:ov]       # slices [transition .. transition+ov-1]

        slice_spacing = self.spacing[0]

        # physical z of slice 0 of each block
        z0_upper = (transition - ov) * slice_spacing
        z0_lower = transition * slice_spacing

        # Convert to vtkImageData in *world* coordinates
        img_upper = self._vtk_image_from_block(upper_ov, z0_mm=z0_upper)
        img_lower = self._vtk_image_from_block(lower_ov, z0_mm=z0_lower)

        def make_surface(img):
            mc = vtk.vtkMarchingCubes()
            mc.SetInputData(img)
            mc.SetValue(0, bone_iso)
            mc.Update()
            poly = mc.GetOutput()

            dec = vtk.vtkDecimatePro()
            dec.SetInputData(poly)
            dec.SetTargetReduction(0.8)
            dec.PreserveTopologyOn()
            dec.Update()
            return dec.GetOutput()

        surf_upper = make_surface(img_upper)
        surf_lower = make_surface(img_lower)

        icp = vtk.vtkIterativeClosestPointTransform()
        icp.SetSource(surf_lower)   # lower -> upper
        icp.SetTarget(surf_upper)
        icp.GetLandmarkTransform().SetModeToRigidBody()
        icp.SetMaximumNumberOfIterations(200)
        icp.StartByMatchingCentroidsOn()
        icp.Modified()
        icp.Update()

        tf = vtk.vtkTransform()
        tf.SetMatrix(icp.GetMatrix())

        print("ICP transform matrix:")
        print(tf.GetMatrix())

        return tf 
    
    def _to_vtk_image(self):
        """Convert self.volume (NumPy) to vtkImageData using self.spacing."""
        import vtk
        from vtk.util import numpy_support as nps

        vol = self.volume.astype(np.int16, copy=False)
        nz, ny, nx = vol.shape

        flat = vol.ravel(order="C")
        vtk_array = nps.numpy_to_vtk(
            num_array=flat,
            deep=True,
            array_type=vtk.VTK_SHORT
        )

        image = vtk.vtkImageData()
        image.SetDimensions(nx, ny, nz)
        image.SetSpacing(self.spacing[2], self.spacing[1], self.spacing[0])
        image.SetOrigin(0.0, 0.0, 0.0)
        image.GetPointData().SetScalars(vtk_array)

        return image
    
    def _aorta_mask_to_vtk_image(self):
        """Convert aorta mask (NumPy) to vtkImageData."""
        import vtk
        from vtk.util import numpy_support as nps
        
        if self.aorta_mask is None:
            return None
        
        # The segmentation from Slicer is stored as (X, Y, Z) based on the NRRD header
        # with sizes [512, 512, 157] and space directions showing X, Y, Z order
        # VTK with C-order flattening expects the numpy array in (Z, Y, X) order
        # So we need to transpose: (X, Y, Z) -> (Z, Y, X)
        mask_transposed = np.transpose(self.aorta_mask, (2, 1, 0))  # Now (Z, Y, X) = (157, 512, 512)
        
        # Flip Z axis because:
        # - In DICOM, Z increases from feet to head
        # - In our VTK setup, Z increases from head to feet (sorted head-to-toe)
        # - VTK data extends from origin in POSITIVE Z direction
        # - So we need to flip so the data extends correctly
        mask_flipped = np.flip(mask_transposed, axis=0)
        
        # Convert mask to int16 for VTK (1000 where aorta, 0 elsewhere)
        # vol = (mask_flipped * 1000).astype(np.int16)
        vol = (mask_flipped.astype(np.int16) * 1000)
        nz, ny, nx = vol.shape  # (157, 512, 512)
        
        flat = vol.ravel(order="C")
        vtk_array = nps.numpy_to_vtk(
            num_array=flat,
            deep=True,
            array_type=vtk.VTK_SHORT
        )
        
        image = vtk.vtkImageData()
        image.SetDimensions(nx, ny, nz)  # (512, 512, 157)
        
        # Spacing: aorta_spacing is [X, Y, Z] = [0.93, 0.93, 5.0]
        image.SetSpacing(self.aorta_spacing[0], self.aorta_spacing[1], self.aorta_spacing[2])
        
        # Compute origin offset relative to CT volume
        # After flipping Z, index 0 now corresponds to the TOP of the aorta volume
        # (highest DICOM Z, lowest VTK Z)
        aorta_z_extent = (nz - 1) * self.aorta_spacing[2]  # Total Z extent in mm
        aorta_top_dicom_z = self.aorta_origin[2] + aorta_z_extent  # Highest DICOM Z of aorta
        
        if self.ct_world_origin is not None:
            # X and Y: direct offset from CT origin
            vtk_origin_x = self.aorta_origin[0] - self.ct_world_origin[0]
            vtk_origin_y = self.aorta_origin[1] - self.ct_world_origin[1]
            # Z: After flipping, origin is at the TOP of aorta (highest DICOM Z)
            # VTK_Z = ct_world_origin[2] - aorta_top_dicom_z
            vtk_origin_z = self.ct_world_origin[2] - aorta_top_dicom_z
            print(f"Aorta DICOM Z range: {self.aorta_origin[2]:.1f} to {aorta_top_dicom_z:.1f}")
            print(f"Aorta VTK origin: ({vtk_origin_x:.1f}, {vtk_origin_y:.1f}, {vtk_origin_z:.1f}) mm")
        else:
            vtk_origin_x = self.aorta_origin[0]
            vtk_origin_y = self.aorta_origin[1]
            vtk_origin_z = -aorta_top_dicom_z
            
        image.SetOrigin(vtk_origin_x, vtk_origin_y, vtk_origin_z)
        image.GetPointData().SetScalars(vtk_array)
        
        return image
    
    def _bone_mask_to_vtk_image(self):
        """Convert bone mask (NumPy labeled array) to vtkImageData, preserving labels."""
        import vtk
        from vtk.util import numpy_support as nps
        
        if self.bone_mask is None:
            return None
        
        # Same transformation as aorta: (X, Y, Z) -> (Z, Y, X)
        mask_transposed = np.transpose(self.bone_mask, (2, 1, 0))
        mask_flipped = np.flip(mask_transposed, axis=0)
        
        # Keep original label values (uint8 to int16)
        vol = mask_flipped.astype(np.int16)
        nz, ny, nx = vol.shape
        
        flat = vol.ravel(order="C")
        vtk_array = nps.numpy_to_vtk(
            num_array=flat,
            deep=True,
            array_type=vtk.VTK_SHORT
        )
        
        image = vtk.vtkImageData()
        image.SetDimensions(nx, ny, nz)
        image.SetSpacing(self.bone_spacing[0], self.bone_spacing[1], self.bone_spacing[2])
        
        # Compute origin offset relative to CT volume (same logic as aorta)
        bone_z_extent = (nz - 1) * self.bone_spacing[2]
        bone_top_dicom_z = self.bone_origin[2] + bone_z_extent
        
        if self.ct_world_origin is not None:
            vtk_origin_x = self.bone_origin[0] - self.ct_world_origin[0]
            vtk_origin_y = self.bone_origin[1] - self.ct_world_origin[1]
            vtk_origin_z = self.ct_world_origin[2] - bone_top_dicom_z
            print(f"Bone DICOM Z range: {self.bone_origin[2]:.1f} to {bone_top_dicom_z:.1f}")
            print(f"Bone VTK origin: ({vtk_origin_x:.1f}, {vtk_origin_y:.1f}, {vtk_origin_z:.1f}) mm")
        else:
            vtk_origin_x = self.bone_origin[0]
            vtk_origin_y = self.bone_origin[1]
            vtk_origin_z = -bone_top_dicom_z
            
        image.SetOrigin(vtk_origin_x, vtk_origin_y, vtk_origin_z)
        image.GetPointData().SetScalars(vtk_array)
        
        return image
    
    def _create_bone_actors_with_labels(self, bone_image, bones_only=True):
        """
        Create individual actors for each labeled bone segment.
        Returns list of (actor, label_value, name, color) tuples.
        
        Args:
            bone_image: vtkImageData with labeled segments
            bones_only: If True, only render skeletal structures (faster)
        """
        import vtk
        
        actors_info = []
        
        if self.bone_labels is None or bone_image is None:
            return actors_info
        
        # Keywords for bones
        bone_keywords = ['rib', 'vertebra', 'sacrum', 'hip', 'femur', 'scapula', 
                        'clavicula', 'clavicle', 'humerus', 'sternum', 'costal', 
                        'iliac', 'bone', 'skull', 'mandible']
        
        # Get unique labels from the data
        from vtk.util import numpy_support as nps
        scalars = bone_image.GetPointData().GetScalars()
        data_array = nps.vtk_to_numpy(scalars)
        unique_labels = np.unique(data_array)
        unique_labels = unique_labels[unique_labels > 0]  # Exclude background
        
        # Filter labels if bones_only
        if bones_only:
            filtered_labels = []
            for label_val in unique_labels:
                label_str = str(label_val)
                if label_str in self.bone_labels:
                    name = self.bone_labels[label_str].get('name', '').lower()
                    if any(kw in name for kw in bone_keywords):
                        filtered_labels.append(label_val)
            unique_labels = filtered_labels
            print(f"\nCreating surfaces for {len(unique_labels)} bone segments (filtering organs)...")
        else:
            print(f"\nCreating surfaces for {len(unique_labels)} segments...")
        
        total = len(unique_labels)
        for idx, label_val in enumerate(unique_labels):
            label_str = str(label_val)
            if label_str not in self.bone_labels:
                continue
            
            label_info = self.bone_labels[label_str]
            name = label_info.get('name', f'Segment {label_val}')
            color = label_info.get('color', [1.0, 1.0, 1.0])
            
            # Progress indicator
            if (idx + 1) % 10 == 0 or idx == total - 1:
                print(f"  Processing {idx + 1}/{total}: {name}")
            
            # Threshold to extract only this label
            threshold = vtk.vtkImageThreshold()
            threshold.SetInputData(bone_image)
            threshold.ThresholdBetween(label_val, label_val)
            threshold.SetInValue(1000)
            threshold.SetOutValue(0)
            threshold.Update()
            
            # Marching cubes to create surface
            mc = vtk.vtkMarchingCubes()
            mc.SetInputConnection(threshold.GetOutputPort())
            mc.SetValue(0, 500)
            mc.Update()
            
            # Skip if no surface generated
            if mc.GetOutput().GetNumberOfPoints() == 0:
                continue
            
            # Decimate to reduce polygon count for performance
            decimate = vtk.vtkDecimatePro()
            decimate.SetInputConnection(mc.GetOutputPort())
            decimate.SetTargetReduction(0.5)  # Reduce by 50%
            decimate.PreserveTopologyOn()
            decimate.Update()
            
            # Smooth the surface
            smoother = vtk.vtkSmoothPolyDataFilter()
            smoother.SetInputConnection(decimate.GetOutputPort())
            smoother.SetNumberOfIterations(15)
            smoother.SetRelaxationFactor(0.1)
            smoother.Update()
            
            # Create mapper
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(smoother.GetOutputPort())
            mapper.ScalarVisibilityOff()
            
            # Create actor
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(color[0], color[1], color[2])
            actor.SetPickable(True)  # Ensure it's pickable
            
            # Check if this is a vein/artery (keep visible) or bone (invisible until hover)
            name_lower = name.lower()
            is_vessel = any(v in name_lower for v in ['aorta', 'vein', 'vena', 'artery', 
                                                       'pulmonary', 'portal', 'hepatic vessel'])
            
            if is_vessel:
                # Vessels stay visible
                actor.GetProperty().SetOpacity(1.0)
                actor.GetProperty().SetAmbient(0.5)
                actor.GetProperty().SetDiffuse(0.7)
                actor.GetProperty().SetSpecular(0.5)
            else:
                # Bones/organs: slightly visible for picking, highlights on hover
                actor.GetProperty().SetOpacity(0.15)  # Visible enough to pick
                actor.GetProperty().SetAmbient(0.2)
                actor.GetProperty().SetDiffuse(0.5)
                actor.GetProperty().SetSpecular(0.3)
            
            actors_info.append((actor, label_val, name, color, is_vessel))
        
        vessel_count = sum(1 for _, _, _, _, is_v in actors_info if is_v)
        bone_count = len(actors_info) - vessel_count
        print(f"✓ Created {len(actors_info)} surfaces ({bone_count} bones, {vessel_count} vessels)")
        return actors_info
    
    def show_vtk_volume_icp_registered(self, bone_iso=300, alignment_threshold_mm=3.0):
        """
        Volume rendering with smart per-leg alignment correction.
        
        1. Detect the two legs at the junction point
        2. Check alignment quality for each leg (compare bone centroids)
        3. Only correct legs that are misaligned (beyond threshold)
        4. Apply correction by shifting voxels in numpy array
        5. Render as single corrected volume
        """
        from scipy import ndimage as ndi

        transition = self._find_series_transition_index()
        if transition is None:
            print("Only one series – using normal volume rendering.")
            self.show_vtk_volume()
            return

        print("="*60)
        print("Smart Leg Alignment")
        print("="*60)
        
        # Work on a copy of the volume
        corrected_volume = self.volume.copy()
        
        pixel_spacing_x = self.spacing[2]  # mm per pixel in X
        pixel_spacing_y = self.spacing[1]  # mm per pixel in Y
        
        # Number of slices to analyze at the junction
        junction_slices = 15
        
        # Get slices just above and below the transition
        upper_junction = corrected_volume[transition - junction_slices:transition]
        lower_junction = corrected_volume[transition:transition + junction_slices]
        
        print(f"Transition at slice: {transition}")
        print(f"Analyzing {junction_slices} slices above and below junction")
        
        # --- Detect legs in upper junction (femurs) ---
        upper_bone = upper_junction > bone_iso
        upper_labeled, upper_num = ndi.label(upper_bone)
        
        if upper_num < 2:
            print("Could not detect two femurs in upper junction - using normal rendering")
            self.show_vtk_volume()
            return
        
        # Find two largest bone components in upper (the femurs)
        upper_sizes = ndi.sum(upper_bone, upper_labeled, range(1, upper_num + 1))
        upper_sorted = np.argsort(upper_sizes)[::-1]
        
        femur1_label = upper_sorted[0] + 1
        femur2_label = upper_sorted[1] + 1
        
        # Get centroids (in the last slice of upper junction - closest to cut)
        upper_last_slice = upper_junction[-1] > bone_iso
        upper_last_labeled, _ = ndi.label(upper_last_slice)
        
        # Find centroids in the last slice
        femur_centroids_upper = []
        for label in [femur1_label, femur2_label]:
            mask = upper_labeled[-1] == label
            if mask.any():
                coords = np.where(mask)
                centroid_y = np.mean(coords[0])
                centroid_x = np.mean(coords[1])
                femur_centroids_upper.append((centroid_y, centroid_x))
        
        if len(femur_centroids_upper) < 2:
            # Try connected components on just the last slice
            upper_last_labeled, num = ndi.label(upper_last_slice)
            if num >= 2:
                sizes = ndi.sum(upper_last_slice, upper_last_labeled, range(1, num + 1))
                sorted_labels = np.argsort(sizes)[::-1]
                femur_centroids_upper = []
                for i in range(2):
                    label = sorted_labels[i] + 1
                    coords = np.where(upper_last_labeled == label)
                    centroid_y = np.mean(coords[0])
                    centroid_x = np.mean(coords[1])
                    femur_centroids_upper.append((centroid_y, centroid_x))
        
        if len(femur_centroids_upper) < 2:
            print("Could not find two femur centroids - using normal rendering")
            self.show_vtk_volume()
            return
        
        # Sort by X coordinate (left vs right)
        femur_centroids_upper.sort(key=lambda c: c[1])
        left_femur_upper = femur_centroids_upper[0]  # smaller X = left
        right_femur_upper = femur_centroids_upper[1]  # larger X = right
        
        print(f"Upper left femur centroid (y,x): ({left_femur_upper[0]:.1f}, {left_femur_upper[1]:.1f})")
        print(f"Upper right femur centroid (y,x): ({right_femur_upper[0]:.1f}, {right_femur_upper[1]:.1f})")
        
        # --- Detect legs in lower junction ---
        lower_bone = lower_junction > bone_iso
        lower_first_slice = lower_junction[0] > bone_iso
        lower_first_labeled, num = ndi.label(lower_first_slice)
        
        if num < 2:
            print("Could not detect two leg bones in lower junction - using normal rendering")
            self.show_vtk_volume()
            return
        
        # Find two largest components
        sizes = ndi.sum(lower_first_slice, lower_first_labeled, range(1, num + 1))
        sorted_labels = np.argsort(sizes)[::-1]
        
        leg_centroids_lower = []
        leg_labels = []
        for i in range(2):
            label = sorted_labels[i] + 1
            coords = np.where(lower_first_labeled == label)
            centroid_y = np.mean(coords[0])
            centroid_x = np.mean(coords[1])
            leg_centroids_lower.append((centroid_y, centroid_x))
            leg_labels.append(label)
        
        # Sort by X coordinate
        sorted_indices = sorted(range(2), key=lambda i: leg_centroids_lower[i][1])
        left_leg_lower = leg_centroids_lower[sorted_indices[0]]
        right_leg_lower = leg_centroids_lower[sorted_indices[1]]
        left_leg_label = leg_labels[sorted_indices[0]]
        right_leg_label = leg_labels[sorted_indices[1]]
        
        print(f"Lower left leg centroid (y,x): ({left_leg_lower[0]:.1f}, {left_leg_lower[1]:.1f})")
        print(f"Lower right leg centroid (y,x): ({right_leg_lower[0]:.1f}, {right_leg_lower[1]:.1f})")
        
        # --- Calculate misalignment for each leg ---
        left_offset_y = left_femur_upper[0] - left_leg_lower[0]
        left_offset_x = left_femur_upper[1] - left_leg_lower[1]
        left_offset_mm = np.sqrt((left_offset_y * pixel_spacing_y)**2 + 
                                  (left_offset_x * pixel_spacing_x)**2)
        
        right_offset_y = right_femur_upper[0] - right_leg_lower[0]
        right_offset_x = right_femur_upper[1] - right_leg_lower[1]
        right_offset_mm = np.sqrt((right_offset_y * pixel_spacing_y)**2 + 
                                   (right_offset_x * pixel_spacing_x)**2)
        
        print(f"\nLeft leg offset: ({left_offset_y:.1f}, {left_offset_x:.1f}) pixels = {left_offset_mm:.1f} mm")
        print(f"Right leg offset: ({right_offset_y:.1f}, {right_offset_x:.1f}) pixels = {right_offset_mm:.1f} mm")
        print(f"Alignment threshold: {alignment_threshold_mm} mm")
        
        # --- Create proper anatomical leg masks using BONE-based 3D connected components ---
        # Using bone (not tissue) prevents the CT table from being grouped with legs
        print("\nCreating anatomical leg masks (bone-based)...")
        
        lower_volume = corrected_volume[transition:]
        
        # Create BONE mask for lower volume (bone won't include table)
        bone_mask = lower_volume > bone_iso
        
        # Label connected components in 3D on bone
        bone_labeled, num_bone_components = ndi.label(bone_mask)
        print(f"  Found {num_bone_components} connected bone regions")
        
        if num_bone_components >= 2:
            # Find component sizes
            component_sizes = ndi.sum(bone_mask, bone_labeled, range(1, num_bone_components + 1))
            sorted_components = np.argsort(component_sizes)[::-1]
            
            # Get the two largest bone components (should be the two leg bones)
            comp1_label = sorted_components[0] + 1
            comp2_label = sorted_components[1] + 1
            
            # Determine which is left and which is right by checking X centroid
            comp1_coords = np.where(bone_labeled == comp1_label)
            comp2_coords = np.where(bone_labeled == comp2_label)
            
            comp1_x_mean = np.mean(comp1_coords[2])  # X is axis 2
            comp2_x_mean = np.mean(comp2_coords[2])
            
            if comp1_x_mean < comp2_x_mean:
                left_bone_mask = (bone_labeled == comp1_label)
                right_bone_mask = (bone_labeled == comp2_label)
            else:
                left_bone_mask = (bone_labeled == comp2_label)
                right_bone_mask = (bone_labeled == comp1_label)
            
            # Dilate bone masks to include surrounding soft tissue
            # This expands the bone mask to capture the full leg
            dilation_iterations = 30  # ~22mm at 0.74mm/pixel
            struct = ndi.generate_binary_structure(3, 1)  # 6-connectivity
            
            left_leg_mask = ndi.binary_dilation(left_bone_mask, struct, iterations=dilation_iterations)
            right_leg_mask = ndi.binary_dilation(right_bone_mask, struct, iterations=dilation_iterations)
            
            # Only keep voxels that are actually tissue (not air)
            tissue_mask = lower_volume > -200
            left_leg_mask = left_leg_mask & tissue_mask
            right_leg_mask = right_leg_mask & tissue_mask
            
            print(f"  Left leg mask: {left_leg_mask.sum()} voxels (from {left_bone_mask.sum()} bone voxels)")
            print(f"  Right leg mask: {right_leg_mask.sum()} voxels (from {right_bone_mask.sum()} bone voxels)")
            use_masks = True
        else:
            print("  Could not separate legs into bone components - using X-coordinate fallback")
            use_masks = False
            center_x = int((left_leg_lower[1] + right_leg_lower[1]) / 2)
        
        # --- Apply corrections only where needed ---
        if left_offset_mm > alignment_threshold_mm:
            print(f"\n→ Left leg needs correction: shifting by ({int(round(left_offset_y))}, {int(round(left_offset_x))}) pixels")
            shift_y = int(round(left_offset_y))
            shift_x = int(round(left_offset_x))
            if use_masks:
                corrected_volume = self._shift_leg_with_mask(
                    corrected_volume, transition, 
                    leg_mask=left_leg_mask,
                    shift_y=shift_y, shift_x=shift_x
                )
            else:
                corrected_volume = self._shift_leg_region(
                    corrected_volume, transition, 
                    x_start=0, x_end=center_x,
                    shift_y=shift_y, shift_x=shift_x
                )
        else:
            print(f"\n✓ Left leg is aligned (within {alignment_threshold_mm} mm)")
        
        if right_offset_mm > alignment_threshold_mm:
            print(f"→ Right leg needs correction: shifting by ({int(round(right_offset_y))}, {int(round(right_offset_x))}) pixels")
            shift_y = int(round(right_offset_y))
            shift_x = int(round(right_offset_x))
            if use_masks:
                corrected_volume = self._shift_leg_with_mask(
                    corrected_volume, transition,
                    leg_mask=right_leg_mask,
                    shift_y=shift_y, shift_x=shift_x
                )
            else:
                corrected_volume = self._shift_leg_region(
                    corrected_volume, transition,
                    x_start=center_x, x_end=corrected_volume.shape[2],
                    shift_y=shift_y, shift_x=shift_x
                )
        else:
            print(f"✓ Right leg is aligned (within {alignment_threshold_mm} mm)")
        
        # --- Render the corrected volume ---
        print("\nRendering corrected volume...")
        
        # Temporarily replace self.volume with corrected version for rendering
        original_volume = self.volume
        self.volume = corrected_volume
        self.show_vtk_volume()
        self.volume = original_volume  # Restore original
    
    def _shift_leg_with_mask(self, volume, transition, leg_mask, shift_y, shift_x):
        """
        Shift voxels belonging to a specific leg (defined by 3D mask) by the given offset.
        Uses proper anatomical mask - shifts the ENTIRE leg as one object.
        
        Args:
            volume: Full volume array
            transition: Slice index where upper/lower volumes meet
            leg_mask: Boolean mask (same shape as lower volume) indicating which voxels belong to this leg
            shift_y, shift_x: Pixel shift amounts
        """
        result = volume.copy()
        lower = volume[transition:].copy()
        nz, ny, nx = lower.shape
        
        # Get the original values at leg positions
        original_lower = volume[transition:]
        
        # Get all coordinates where the leg exists
        leg_coords = np.where(leg_mask)
        leg_values = original_lower[leg_coords]
        
        # Clear the original leg region in result (set to air)
        result[transition:][leg_mask] = -1000
        
        # Calculate new coordinates after shift
        new_z = leg_coords[0]  # Z doesn't change
        new_y = leg_coords[1] + shift_y
        new_x = leg_coords[2] + shift_x
        
        # Filter out coordinates that would be out of bounds
        valid = (new_y >= 0) & (new_y < ny) & (new_x >= 0) & (new_x < nx)
        
        # Apply the shift - place voxels at new locations
        result[transition:][new_z[valid], new_y[valid], new_x[valid]] = leg_values[valid]
        
        return result
    
    def _shift_leg_region(self, volume, transition, x_start, x_end, shift_y, shift_x):
        """
        Fallback: Shift voxels in a leg region using X-coordinate split.
        Only affects the lower part of the volume (below transition).
        """
        # Extract the lower portion that needs shifting
        lower = volume[transition:].copy()
        nz, ny, nx = lower.shape
        
        # Create output array filled with air
        shifted_lower = np.full_like(lower, -1000)
        
        # Calculate source and destination ranges for the shift
        # Source range in Y
        if shift_y >= 0:
            src_y_start = 0
            src_y_end = ny - shift_y
            dst_y_start = shift_y
            dst_y_end = ny
        else:
            src_y_start = -shift_y
            src_y_end = ny
            dst_y_start = 0
            dst_y_end = ny + shift_y
        
        # Source range in X (within the leg region)
        if shift_x >= 0:
            src_x_start = x_start
            src_x_end = min(x_end - shift_x, nx)
            dst_x_start = x_start + shift_x
            dst_x_end = min(x_end, nx)
        else:
            src_x_start = max(x_start - shift_x, 0)
            src_x_end = x_end
            dst_x_start = x_start
            dst_x_end = x_end + shift_x
        
        # Ensure valid ranges
        if src_y_end > src_y_start and src_x_end > src_x_start:
            if dst_y_end > dst_y_start and dst_x_end > dst_x_start:
                # Copy the shifted leg region
                shifted_lower[:, dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
                    lower[:, src_y_start:src_y_end, src_x_start:src_x_end]
        
        # Copy the OTHER leg region unchanged
        if x_start == 0:
            # We shifted left leg, copy right leg unchanged
            shifted_lower[:, :, x_end:] = lower[:, :, x_end:]
        else:
            # We shifted right leg, copy left leg unchanged
            shifted_lower[:, :, :x_start] = lower[:, :, :x_start]
        
        # Put back into volume
        result = volume.copy()
        result[transition:] = shifted_lower
        
        return result
        
    def show_vtk_volume(self):
        """Interactive VTK volume rendering using CT-style transfer functions."""
        import vtk

        image_data = self._to_vtk_image()

        ren = vtk.vtkRenderer()
        ren.SetBackground(0.2, 0.2, 0.2)
        ren.SetBackground2(0.5, 0.5, 0.5)
        ren.GradientBackgroundOn()

        renWin = vtk.vtkRenderWindow()
        renWin.AddRenderer(ren)
        renWin.SetSize(800, 800)

        iren = vtk.vtkRenderWindowInteractor()
        iren.SetRenderWindow(renWin)

        volumeMapper = vtk.vtkGPUVolumeRayCastMapper()
        volumeMapper.SetInputData(image_data)
        volumeMapper.SetBlendModeToComposite()

        smin, smax = image_data.GetScalarRange()
        print("Scalar range:", smin, smax)

        volumeCTF = vtk.vtkColorTransferFunction()
        volumeCTF.AddRGBPoint(-1000, 0.0, 0.0, 0.0)
        volumeCTF.AddRGBPoint(-200,  0.4, 0.2, 0.2)
        volumeCTF.AddRGBPoint( 300,  1.0, 0.8, 0.6)
        volumeCTF.AddRGBPoint(1000,  1.0, 1.0, 0.9)

        volumeOTF = vtk.vtkPiecewiseFunction()
        # volumeOTF.AddPoint(-1000, 0.00)
        # volumeOTF.AddPoint(-200,  0.00)
        # volumeOTF.AddPoint( 200,  0.15)
        # volumeOTF.AddPoint( 700,  0.35)
        # volumeOTF.AddPoint(1000,  0.85)
        volumeOTF.AddPoint(-1000, 0.0)   # air
        volumeOTF.AddPoint(-200,  0.0)   # fat
        volumeOTF.AddPoint(   0,  0.01)  # very faint tissue
        volumeOTF.AddPoint( 150, 0.03)
        volumeOTF.AddPoint( 250, 0.06)
        volumeOTF.AddPoint( 400, 0.25)   # start of bone
        volumeOTF.AddPoint( 700, 0.6)
        volumeOTF.AddPoint(1000, 0.9)   

        volumeGOTF = vtk.vtkPiecewiseFunction()
        volumeGOTF.AddPoint(0,   0.0)
        volumeGOTF.AddPoint(90,  0.5)
        volumeGOTF.AddPoint(100, 1.0)

        volumeProperty = vtk.vtkVolumeProperty()
        volumeProperty.SetColor(volumeCTF)
        volumeProperty.SetScalarOpacity(volumeOTF)
        volumeProperty.SetGradientOpacity(volumeGOTF)
        volumeProperty.SetInterpolationTypeToLinear()
        volumeProperty.ShadeOn()
        volumeProperty.SetAmbient(0.3)
        volumeProperty.SetDiffuse(0.6)
        volumeProperty.SetSpecular(0.2)

        volume = vtk.vtkVolume()
        volume.SetMapper(volumeMapper)
        volume.SetProperty(volumeProperty)

        ren.AddViewProp(volume)
        
        # Add aorta segmentation as a red surface if available
        if self.aorta_mask is not None:
            print("\nRendering aorta segmentation...")
            aorta_image = self._aorta_mask_to_vtk_image()
            
            # Create marching cubes to extract aorta surface
            mc = vtk.vtkMarchingCubes()
            mc.SetInputData(aorta_image)
            mc.SetValue(0, 500)  # Threshold at 500 (mask has values of 1000)
            mc.Update()
            
            # Smooth the surface
            smoother = vtk.vtkSmoothPolyDataFilter()
            smoother.SetInputConnection(mc.GetOutputPort())
            smoother.SetNumberOfIterations(50)
            smoother.SetRelaxationFactor(0.1)
            smoother.Update()
            
            # Create mapper for the aorta surface
            aorta_mapper = vtk.vtkPolyDataMapper()
            aorta_mapper.SetInputConnection(smoother.GetOutputPort())
            aorta_mapper.ScalarVisibilityOff()
            
            # Create actor for the aorta
            aorta_actor = vtk.vtkActor()
            aorta_actor.SetMapper(aorta_mapper)
            aorta_actor.GetProperty().SetColor(1.0, 0.0, 0.0)  # Bright RED
            aorta_actor.GetProperty().SetOpacity(1.0)  # Fully opaque
            aorta_actor.GetProperty().SetAmbient(0.5)  # High ambient so always visible
            aorta_actor.GetProperty().SetDiffuse(0.7)
            aorta_actor.GetProperty().SetSpecular(0.5)
            aorta_actor.GetProperty().SetSpecularPower(30)
            
            ren.AddActor(aorta_actor)
            print("✓ Aorta segmentation added as red surface")

        # Add bone segmentation with labeled surfaces and hover tooltip
        bone_actors_info = []
        if self.bone_mask is not None:
            print("\nRendering bone segmentation with labels...")
            bone_image = self._bone_mask_to_vtk_image()
            bone_actors_info = self._create_bone_actors_with_labels(bone_image)
            
            for actor, label_val, name, color, is_vessel in bone_actors_info:
                ren.AddActor(actor)
            
            print(f"✓ Added {len(bone_actors_info)} labeled bone/organ surfaces")
        else:
            print("\n⚠️  No bone segmentation loaded - hover labels disabled")
        
        # Add aorta to the hover detection list (if it exists) - it's a vessel so stays visible
        if self.aorta_mask is not None:
            bone_actors_info.append((aorta_actor, -1, "Aorta", [1.0, 0.0, 0.0], True))
        
        c = volume.GetCenter()
        cam = ren.GetActiveCamera()
        cam.SetFocalPoint(c)
        cam.SetPosition(c[0] + 400, c[1], c[2])
        cam.SetViewUp(0, 0, -1)

        # Set up hover tooltip with highlighting for bone/aorta labels
        if bone_actors_info:
            # Create a text actor for displaying labels
            text_actor = vtk.vtkTextActor()
            text_actor.SetInput("")
            text_actor.GetTextProperty().SetFontSize(22)
            text_actor.GetTextProperty().SetColor(1.0, 1.0, 1.0)
            text_actor.GetTextProperty().SetBackgroundColor(0.0, 0.0, 0.0)
            text_actor.GetTextProperty().SetBackgroundOpacity(0.8)
            text_actor.GetTextProperty().BoldOn()
            text_actor.SetPosition(10, 10)
            ren.AddActor2D(text_actor)
            
            # Create picker - CellPicker with precise tolerance
            picker = vtk.vtkCellPicker()
            picker.SetTolerance(0.001)  # Precise picking
            
            # Add ONLY segmentation actors to pick list (ignore volume)
            for actor, label_val, name, color, is_vessel in bone_actors_info:
                picker.AddPickList(actor)
            picker.PickFromListOn()
            
            # Map actors to their info and store original state
            actor_to_info = {}
            actor_original_state = {}
            for actor, label_val, name, color, is_vessel in bone_actors_info:
                actor_to_info[actor] = (label_val, name, color, is_vessel)
                actor_original_state[actor] = {
                    'color': color,
                    'opacity': actor.GetProperty().GetOpacity(),
                    'is_vessel': is_vessel
                }
            
            # Track currently highlighted actor for debouncing
            current_highlight = [None]
            
            def reset_highlight(actor):
                """Reset an actor to its original state"""
                if actor is not None and actor in actor_original_state:
                    state = actor_original_state[actor]
                    actor.GetProperty().SetOpacity(state['opacity'])
                    color = state['color']
                    actor.GetProperty().SetColor(color[0], color[1], color[2])
                    if state['is_vessel']:
                        actor.GetProperty().SetAmbient(0.5)
                        actor.GetProperty().SetSpecular(0.5)
                    else:
                        actor.GetProperty().SetAmbient(0.2)
                        actor.GetProperty().SetSpecular(0.3)
            
            def apply_highlight(actor, color):
                """Apply bright highlight to an actor"""
                actor.GetProperty().SetOpacity(1.0)
                actor.GetProperty().SetAmbient(1.0)
                actor.GetProperty().SetSpecular(1.0)
                highlight_color = [min(1.0, c * 0.5 + 0.5) for c in color]
                actor.GetProperty().SetColor(highlight_color[0], highlight_color[1], highlight_color[2])
            
            # Mouse move callback - simple debouncing
            def on_mouse_move(obj, event):
                x, y = obj.GetEventPosition()
                
                picker.Pick(x, y, 0, ren)
                picked_actor = picker.GetActor()
                
                # Only picked actors in our list count
                if picked_actor not in actor_to_info:
                    picked_actor = None
                
                # Skip if same as current (debounce)
                if picked_actor == current_highlight[0]:
                    return
                
                # Reset previous highlight
                reset_highlight(current_highlight[0])
                
                # Apply new highlight
                if picked_actor is not None:
                    label_val, name, color, is_vessel = actor_to_info[picked_actor]
                    text_actor.SetInput(f" {name} ")
                    text_actor.GetTextProperty().SetColor(color[0], color[1], color[2])
                    apply_highlight(picked_actor, color)
                else:
                    text_actor.SetInput("")
                
                current_highlight[0] = picked_actor
                renWin.Render()
            
            iren.AddObserver("MouseMoveEvent", on_mouse_move)
            print("✓ Hover segmentation enabled - move mouse over anatomy to highlight and see labels")

        renWin.Render()
        iren.Initialize()
        iren.Start()

    def show_vtk_bones(self):
        import vtk

        image_data = self._to_vtk_image()

        ren = vtk.vtkRenderer()
        ren.SetBackground(0.1, 0.1, 0.1)
        renWin = vtk.vtkRenderWindow()
        renWin.AddRenderer(ren)
        renWin.SetSize(800, 800)

        iren = vtk.vtkRenderWindowInteractor()
        iren.SetRenderWindow(renWin)

        volumeMapper = vtk.vtkGPUVolumeRayCastMapper()
        volumeMapper.SetInputData(image_data)
        volumeMapper.SetBlendModeToComposite()

        # --- 🔥 bones-only transfer functions ----

        volumeCTF = vtk.vtkColorTransferFunction()
        # Below ~250 HU: black (won't matter, opacity=0 anyway)
        volumeCTF.AddRGBPoint(-1000, 0.0, 0.0, 0.0)
        volumeCTF.AddRGBPoint(  250, 0.0, 0.0, 0.0)
        # Bone: bright / off-white
        volumeCTF.AddRGBPoint(  400, 0.9, 0.9, 0.9)
        volumeCTF.AddRGBPoint( 1000, 1.0, 1.0, 1.0)
        volumeCTF.AddRGBPoint( 2000, 1.0, 1.0, 1.0)

        volumeOTF = vtk.vtkPiecewiseFunction()
        volumeOTF.AddPoint(-1000, 0.0)
        volumeOTF.AddPoint(  150, 0.0)   # new cutoff ~150 HU
        volumeOTF.AddPoint(  250, 0.15)
        volumeOTF.AddPoint(  400, 0.4)
        volumeOTF.AddPoint(  700, 0.7)
        volumeOTF.AddPoint( 1000, 1.0)

        # Optional: gradient opacity to emphasize edges
        volumeGOTF = vtk.vtkPiecewiseFunction()
        volumeGOTF.AddPoint(0,   0.0)
        volumeGOTF.AddPoint(50,  0.4)
        volumeGOTF.AddPoint(100, 1.0)

        volumeProperty = vtk.vtkVolumeProperty()
        volumeProperty.SetColor(volumeCTF)
        volumeProperty.SetScalarOpacity(volumeOTF)
        volumeProperty.SetGradientOpacity(volumeGOTF)
        volumeProperty.SetInterpolationTypeToLinear()
        volumeProperty.ShadeOn()
        volumeProperty.SetAmbient(0.2)
        volumeProperty.SetDiffuse(0.8)
        volumeProperty.SetSpecular(0.3)

        volume = vtk.vtkVolume()
        volume.SetMapper(volumeMapper)
        volume.SetProperty(volumeProperty)
        ren.AddViewProp(volume)
        
        # Add aorta segmentation as a red surface if available
        if self.aorta_mask is not None:
            print("\nRendering aorta segmentation...")
            aorta_image = self._aorta_mask_to_vtk_image()
            
            # Create marching cubes to extract aorta surface
            mc = vtk.vtkMarchingCubes()
            mc.SetInputData(aorta_image)
            mc.SetValue(0, 500)  # Threshold at 500 (mask has values of 1000)
            mc.Update()
            
            # Smooth the surface
            smoother = vtk.vtkSmoothPolyDataFilter()
            smoother.SetInputConnection(mc.GetOutputPort())
            smoother.SetNumberOfIterations(50)
            smoother.SetRelaxationFactor(0.1)
            smoother.Update()
            
            # Create mapper for the aorta surface
            aorta_mapper = vtk.vtkPolyDataMapper()
            aorta_mapper.SetInputConnection(smoother.GetOutputPort())
            aorta_mapper.ScalarVisibilityOff()
            
            # Create actor for the aorta
            aorta_actor = vtk.vtkActor()
            aorta_actor.SetMapper(aorta_mapper)
            aorta_actor.GetProperty().SetColor(1.0, 0.0, 0.0)  # Bright RED
            aorta_actor.GetProperty().SetOpacity(1.0)  # Fully opaque
            aorta_actor.GetProperty().SetAmbient(0.5)  # High ambient so always visible
            aorta_actor.GetProperty().SetDiffuse(0.7)
            aorta_actor.GetProperty().SetSpecular(0.5)
            aorta_actor.GetProperty().SetSpecularPower(30)
            
            ren.AddActor(aorta_actor)
            print("✓ Aorta segmentation added as red surface")

        c = volume.GetCenter()
        cam = ren.GetActiveCamera()
        cam.SetFocalPoint(c)
        cam.SetPosition(c[0] + 400, c[1], c[2])
        cam.SetViewUp(0, 0, -1)

        renWin.Render()
        iren.Initialize()
        iren.Start()

    def show_vtk_bone_volume(self, threshold=300, min_size=5000):
        """
        Segment bones and render skeleton using VTK.
        """
        import vtk
        import numpy as np
        from vtk.util import numpy_support as nps

        # 1. Segment bones
        bone_mask = self.segment_bone_volume(low_thr=150,
                                        high_thr=300,
                                        min_low_size=500)
        
        # Convert mask to int16 (VTK requires numeric scalars)
        # bone_data = (bone_mask.astype(np.int16) * 1000)  # make bone bright
        # keep original HU for bone, air elsewhere
        bone_data = np.where(bone_mask, self.volume, -1000).astype(np.int16)

        # 2. Wrap mask into vtkImageData
        nz, ny, nx = bone_data.shape
        flat = bone_data.ravel(order="C")
        vtk_array = nps.numpy_to_vtk(flat, deep=True, array_type=vtk.VTK_SHORT)

        image = vtk.vtkImageData()
        image.SetDimensions(nx, ny, nz)
        image.SetSpacing(self.spacing[2], self.spacing[1], self.spacing[0])
        image.GetPointData().SetScalars(vtk_array)

        # 3. VTK renderer (same as before but simpler)
        ren = vtk.vtkRenderer()
        ren.SetBackground(0.1, 0.1, 0.1)
        renWin = vtk.vtkRenderWindow()
        renWin.AddRenderer(ren)
        renWin.SetSize(800, 800)
        iren = vtk.vtkRenderWindowInteractor()
        iren.SetRenderWindow(renWin)

        # 4. GPU mapper
        volumeMapper = vtk.vtkGPUVolumeRayCastMapper()
        volumeMapper.SetInputData(image)
        volumeMapper.SetBlendModeToComposite()

        # 5. transfer functions (binary: bone or nothing)
        volumeCTF = vtk.vtkColorTransferFunction()
        volumeCTF.AddRGBPoint(0, 0, 0, 0)
        volumeCTF.AddRGBPoint(1000, 1, 1, 1)

        volumeOTF = vtk.vtkPiecewiseFunction()
        volumeOTF.AddPoint(0, 0.0)
        volumeOTF.AddPoint(1000, 1.0)

        volumeProperty = vtk.vtkVolumeProperty()
        volumeProperty.SetColor(volumeCTF)
        volumeProperty.SetScalarOpacity(volumeOTF)
        volumeProperty.SetInterpolationTypeToLinear()
        volumeProperty.ShadeOn()
        volumeProperty.SetAmbient(0.3)
        volumeProperty.SetDiffuse(0.8)
        volumeProperty.SetSpecular(0.3)

        volume = vtk.vtkVolume()
        volume.SetMapper(volumeMapper)
        volume.SetProperty(volumeProperty)
        ren.AddVolume(volume)

        # camera
        c = volume.GetCenter()
        cam = ren.GetActiveCamera()
        cam.SetFocalPoint(c)
        cam.SetPosition(c[0] + 400, c[1], c[2])
        cam.SetViewUp(0, 0, -1)

        renWin.Render()
        iren.Initialize()
        iren.Start()

    def show_orthogonal_views(self):
        """Show axial, sagittal, and coronal views with interactive sliders"""
        mid_axial = self.volume.shape[0] // 2
        mid_sagittal = self.volume.shape[1] // 2
        mid_coronal = self.volume.shape[2] // 2
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        plt.subplots_adjust(bottom=0.25)
        
        # Get window settings
        ds = self.datasets[0]
        if hasattr(ds, 'WindowCenter') and hasattr(ds, 'WindowWidth'):
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
            vmin, vmax = -1000, 3000
        
        # Initial images
        img_axial = axes[0, 0].imshow(self.volume[mid_axial, :, :], cmap='gray', vmin=vmin, vmax=vmax)
        axes[0, 0].set_title('Axial View (Top-Down)')
        axes[0, 0].axis('off')
        
        img_sagittal = axes[0, 1].imshow(self.volume[:, mid_sagittal, :], cmap='gray', vmin=vmin, vmax=vmax, aspect=self.spacing[0]/self.spacing[2])
        axes[0, 1].set_title('Sagittal View (Side)')
        axes[0, 1].axis('off')
        
        img_coronal = axes[1, 0].imshow(self.volume[:, :, mid_coronal], cmap='gray', vmin=vmin, vmax=vmax, aspect=self.spacing[0]/self.spacing[1])
        axes[1, 0].set_title('Coronal View (Front)')
        axes[1, 0].axis('off')
        
        # 3D position indicator
        axes[1, 1].text(0.5, 0.5, 'Use sliders below\nto navigate', 
                       ha='center', va='center', fontsize=14)
        axes[1, 1].axis('off')
        
        # Create sliders
        ax_axial = plt.axes([0.1, 0.15, 0.8, 0.03])
        ax_sagittal = plt.axes([0.1, 0.10, 0.8, 0.03])
        ax_coronal = plt.axes([0.1, 0.05, 0.8, 0.03])
        
        slider_axial = Slider(ax_axial, 'Axial', 0, self.volume.shape[0] - 1, valinit=mid_axial, valstep=1)
        slider_sagittal = Slider(ax_sagittal, 'Sagittal', 0, self.volume.shape[1] - 1, valinit=mid_sagittal, valstep=1)
        slider_coronal = Slider(ax_coronal, 'Coronal', 0, self.volume.shape[2] - 1, valinit=mid_coronal, valstep=1)
        
        def update(val):
            img_axial.set_data(self.volume[int(slider_axial.val), :, :])
            img_sagittal.set_data(self.volume[:, int(slider_sagittal.val), :])
            img_coronal.set_data(self.volume[:, :, int(slider_coronal.val)])
            fig.canvas.draw_idle()
        
        slider_axial.on_changed(update)
        slider_sagittal.on_changed(update)
        slider_coronal.on_changed(update)
        
        plt.show()

    
    def show_3d_volume_mip(self):
        """Show Maximum Intensity Projection (MIP) from different angles"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # MIP along each axis
        mip_axial = np.max(self.volume, axis=0)
        mip_sagittal = np.max(self.volume, axis=1)
        mip_coronal = np.max(self.volume, axis=2)
        
        axes[0].imshow(mip_axial, cmap='gray')
        axes[0].set_title('MIP - Axial (Looking Down)')
        axes[0].axis('off')
        
        axes[1].imshow(mip_sagittal, cmap='gray', aspect=self.spacing[0]/self.spacing[2])
        axes[1].set_title('MIP - Sagittal (Looking from Side)')
        axes[1].axis('off')
        
        axes[2].imshow(mip_coronal, cmap='gray', aspect=self.spacing[0]/self.spacing[1])
        axes[2].set_title('MIP - Coronal (Looking from Front)')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def show_3d_interactive(self, threshold=300):
        """Show interactive 3D volume rendering using Plotly"""
        print(f"Creating 3D visualization with threshold={threshold}...")
        print("This may take a moment...")
        
        # Downsample for performance if volume is too large
        step = 2  # Take every 2nd voxel
        volume_small = self.volume[::step, ::step, ::step]
        
        # Create coordinate grids scaled by physical spacing (in mm)
        # X corresponds to slice number (Z direction - head to toe)
        # Y and Z correspond to rows and columns within each slice
        X, Y, Z = np.mgrid[0:volume_small.shape[0], 
                           0:volume_small.shape[1], 
                           0:volume_small.shape[2]]
        
        # Scale by physical spacing to get correct aspect ratio
        X = X * (self.spacing[0] * step)  # Slice spacing
        Y = Y * (self.spacing[1] * step)  # Row spacing
        Z = Z * (self.spacing[2] * step)  # Column spacing
        
        # Flatten arrays
        values = volume_small.flatten()
        
        # Filter by threshold (show only bone/high density structures)
        mask = values > threshold
        
        print(f"Rendering {mask.sum()} voxels above threshold...")
        
        fig = go.Figure(data=go.Volume(
            x=X.flatten()[mask],
            y=Y.flatten()[mask],
            z=Z.flatten()[mask],
            value=values[mask],
            isomin=threshold,
            isomax=values[mask].max(),
            opacity=0.1,
            surface_count=15,
            colorscale='Gray',
            caps=dict(x_show=False, y_show=False, z_show=False),
        ))
        
        fig.update_layout(
            title='3D CT Volume (Interactive) - Physical Dimensions',
            scene=dict(
                xaxis_title='Head-to-Toe (mm)',
                yaxis_title='Left-Right (mm)',
                zaxis_title='Front-Back (mm)',
                aspectmode='data',
            ),
            width=900,
            height=900,
        )
        
        fig.show()
    
    def show_3d_surface(self, threshold=300):
        """Show 3D surface rendering using isosurface"""
        print(f"Creating 3D surface with threshold={threshold}...")
        
        # Downsample
        step = 2
        volume_small = self.volume[::step, ::step, ::step]
        
        # Create isosurface
        from skimage import measure
        try:
            print("Generating isosurface...")
            verts, faces, normals, values = measure.marching_cubes(volume_small, threshold)
            
            print(f"Generated surface with {len(verts)} vertices and {len(faces)} faces")
            
            # Scale vertices by physical spacing to get correct aspect ratio
            verts_scaled = verts.copy()
            verts_scaled[:, 0] *= (self.spacing[0] * step)  # Slice spacing (head-to-toe)
            verts_scaled[:, 1] *= (self.spacing[1] * step)  # Row spacing
            verts_scaled[:, 2] *= (self.spacing[2] * step)  # Column spacing
            
            # Create 3D mesh
            fig = go.Figure(data=[go.Mesh3d(
                x=verts_scaled[:, 0],
                y=verts_scaled[:, 1],
                z=verts_scaled[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color='lightgray',
                opacity=0.8,
                lighting=dict(ambient=0.5, diffuse=0.8, specular=0.5),
                lightposition=dict(x=100, y=100, z=100)
            )])
            
            fig.update_layout(
                title='3D CT Surface Reconstruction - Physical Dimensions',
                scene=dict(
                    xaxis_title='Head-to-Toe (mm)',
                    yaxis_title='Left-Right (mm)',
                    zaxis_title='Front-Back (mm)',
                    aspectmode='data',
                ),
                width=900,
                height=900,
            )
            
            fig.show()
            
        except ImportError:
            print("scikit-image not installed. Installing...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-image"])
            print("Please run this function again.")


def main():
    # Get the DICOM directory
    script_dir = Path(__file__).parent
    dicom_dir = script_dir / "DICOM"
    
    if not dicom_dir.exists():
        print(f"Error: DICOM directory not found at {dicom_dir}")
        sys.exit(1)
    
    # Check for aorta segmentation
    aorta_seg_path = script_dir / "aorta_segmentation_0.npy"
    if not aorta_seg_path.exists():
        aorta_seg_path = None
        print("Note: Aorta segmentation not found, will display CT only")
    
    # Create viewer
    viewer = DICOM3DViewer(dicom_dir, aorta_seg_path=aorta_seg_path)
    
    print("\n" + "="*60)
    print("3D DICOM Viewer - Select visualization mode:")
    print("="*60)
    print("1. Orthogonal Views (Axial/Sagittal/Coronal with sliders)")
    print("2. Maximum Intensity Projection (MIP)")
    print("3. Interactive 3D Volume Rendering")
    print("4. 3D Surface Reconstruction (Isosurface)")
    print("5. All of the above")
    print("="*60)
    
    choice = input("\nEnter your choice (1-5): ").strip()
    
    if choice == '1':
        viewer.show_orthogonal_views()
    elif choice == '2':
        viewer.show_3d_volume_mip()
    elif choice == '3':
        threshold = input("Enter threshold value (default 300 for bone): ").strip()
        threshold = int(threshold) if threshold else 300
        viewer.show_3d_interactive(threshold)
    elif choice == '4':
        threshold = input("Enter threshold value (default 300 for bone): ").strip()
        threshold = int(threshold) if threshold else 300
        viewer.show_3d_surface(threshold)
    elif choice == '5':
        viewer.show_orthogonal_views()
        viewer.show_3d_volume_mip()
        threshold = input("\nEnter threshold value for 3D visualization (default 300 for bone): ").strip()
        threshold = int(threshold) if threshold else 300
        viewer.show_3d_interactive(threshold)
        viewer.show_3d_surface(threshold)
    else:
        print("Invalid choice. Showing orthogonal views...")
        viewer.show_orthogonal_views()


if __name__ == "__main__":
    main()

