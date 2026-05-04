# Pattern Lasso Tool

Interactive tool for extracting sewing pattern panels from images and exporting them into a structured format for downstream processing (e.g., `.ply` meshes for GarmentCode-style datasets).

## Features

- Magnetic lasso / edge-snapping selection
- Panel extraction from raster sewing patterns
- Export to `.ply` mesh format
- Compatible with downstream pattern → mesh → ML pipeline

## Requirements

- Python 3.9+
- PyQt6
- numpy
- opencv-python
- mapbox_earcut

Install:

```bash
pip install PyQt6 numpy opencv-python mapbox_earcut

## How to Use

To use the lasso tool, run the following command:  
```bash
python pattern_lasso.py

#### INSERT IMAGE HERE

Then, use these controls to select the pattern pieces:

| Key  | Action |
| ------------- | ------------- |
| Left Click  | Add anchor point  |
| Right Click  | Undo last anchor  |
| Enter | Close Panel  |
| N  | Save Panel  |
| D  | Duplicate Panel  |
| M  | Mirror Panel (Horizontal)  |
| F  | Flip Panel (Vertical)  |
| E  | Export Current Panel  |
| X  | Export All Saved Panels |
| R  | Reset Current Selection | 


### GUIDE HERE:

limitations on patterns -- what parts should you not select for pants / shirts / jackets ?
pants - button fly / pockets
shirts - pockets
jackets - 



