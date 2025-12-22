# Mass Graves CT Scan Visualization Project

A 3D DICOM viewer for visualizing and analyzing CT scans from archaeological mass grave excavations. Built for forensic research and documentation purposes.

## Important: Data Not Included

**All personal and sensitive data has been removed from this repository.** This includes all DICOM scans, medical imaging files, and any identifying information.

To use this project, you'll need to provide your own DICOM data. Place the files in the appropriate body folders (see structure below).

> **For Dr. Belleman:** If you don't have access to the original datasets, please contact us and we can arrange secure data transfer.

## Quick Start

```bash
cd MassGravesProject/src
python run_viewer.py
```

Select a body dataset from the menu, and the 3D volume renderer will launch. From there you can explore orthogonal views, MIP projections, and 3D surface reconstructions.

## Project Structure

```
MassGravesProject/
├── src/                    # Main application code
│   ├── run_viewer.py       # Entry point - run this
│   └── dicom_viewer_3d.py  # Core 3D viewer (VTK-based)
│
├── data/
│   └── segmentations/      # Pre-computed segmentation overlays
│
├── 2021.003/ - 2021.022/   # Body datasets (add DICOM data here)
│
├── utils/                  # One-time utility scripts
│   ├── convert_nrrd_to_npy.py
│   ├── extract_upper_body.py
│   └── extract_lower_body.py
│
├── legacy/                 # Old/experimental viewers
│
└── requirements.txt        # Python dependencies
```

## Body Datasets

Each body has its own folder. To use a dataset, place the original CD/DVD contents in the corresponding folder:

| Folder        | Description               |
| ------------- | ------------------------- |
| `2021.003/` | Body 1 - Original dataset |
| `2021.013/` | Body 2                    |
| `2021.014/` | Body 3                    |
| `2021.020/` | Body 4                    |
| `2021.021/` | Body 5                    |
| `2021.022/` | Body 6                    |

Each folder should contain a `DICOM/` subfolder with the raw scan files.

## Features

- **3D Volume Rendering** - VTK-based interactive volume visualization
- **Orthogonal Views** - Axial, sagittal, and coronal slice navigation
- **Maximum Intensity Projection (MIP)** - For vascular and bone visualization
- **3D Surface Reconstruction** - Isosurface extraction for 3D models
- **Segmentation Overlays** - Aorta and bone highlighting (Body 1 only)

## Requirements

- Python 3.8+
- See `requirements.txt` for dependencies

```bash
pip install -r MassGravesProject/requirements.txt
```

## Notes

- DICOM data is not included in the repository (too large, and contains sensitive medical imaging)
- Segmentation data was generated using MONAI
- The `.nrrd` files in `DICOM_upper_body/` were used as input for the segmentation pipeline
