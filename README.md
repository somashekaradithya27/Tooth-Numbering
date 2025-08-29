# Tooth-Numbering

## Overview

The Tooth-Numbering project aims to detect and number teeth in dental images using object detection with the Ultralytics YOLOv8 model. The workflow includes preparing a labeled dataset (FDI notation), training a YOLOv8 model, and evaluating its performance for automated dental charting.

## Features

- Dataset preparation from raw zipped data containing dental images and labels.
- Cleaning and validation of label files for YOLO format.
- Automatic dataset split into train/val/test sets.
- Training and evaluating YOLOv8 models using the Ultralytics library.
- Custom class names for FDI tooth numbering notation.
- Prediction and visualization of results on validation images.

## Getting Started

### 1. Environment Setup

You need Python 3.12+ and a GPU-enabled environment for training.
Install required packages:

```bash
pip install "ultralytics>=8.3.0,<9"
```

### 2. Data Preparation

- Upload your zipped dataset (e.g., `ToothNumber_Raw.zip`) containing an `images/` and `labels/` folder in Colab or your environment.
- Extract the zip and validate the folder structure:

```
ToothNumber_TaskDataset/
  ├── images/
  └── labels/
```

- The notebook automatically cleans label files to ensure each label line contains:  
  `[class_id cx cy w h]` – where coordinates are normalized.

- Dataset is split into 80% train, 10% val, 10% test.

### 3. Dataset Structure

After running the notebook, your dataset will be structured as:

```
ToothNumber_TaskDataset/
  ├── images/
  │   ├── train/
  │   ├── val/
  │   └── test/
  └── labels/
      ├── train/
      ├── val/
      └── test/
data.yaml           # dataset configuration for YOLO
```

FDI tooth classes (32 total) are listed in `data.yaml`.

### 4. Training the Model

Train YOLOv8 using the prepared data:

```python
from ultralytics import YOLO

model = YOLO('yolov8s.pt')
results = model.train(
    data='data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    workers=2,
    device=0,          # 0 for GPU
    cache=True,
    fliplr=0.0, flipud=0.0,   # No flips for FDI notation
    mosaic=0.1, mixup=0.0,
    name='dental_fdi_pretrained',
    project='runs'
)
```

### 5. Evaluation & Prediction

Evaluate on validation and test sets:

```python
model.val(data='data.yaml', imgsz=640)                 # val
model.val(data='data.yaml', split='test', imgsz=640)   # test
```

Generate predictions for reporting:

```python
model.predict(source='ToothNumber_TaskDataset/images/val', imgsz=640, conf=0.25, save=True)
```

## FDI Tooth Classes

The project uses these FDI tooth classes:

```
Canine (13), Canine (23), Canine (33), Canine (43),
Central Incisor (21), Central Incisor (41), Central Incisor (31), Central Incisor (11),
First Molar (16), First Molar (26), First Molar (36), First Molar (46),
First Premolar (14), First Premolar (34), First Premolar (44), First Premolar (24),
Lateral Incisor (22), Lateral Incisor (32), Lateral Incisor (42), Lateral Incisor (12),
Second Molar (17), Second Molar (27), Second Molar (37), Second Molar (47),
Second Premolar (15), Second Premolar (25), Second Premolar (35), Second Premolar (45),
Third Molar (18), Third Molar (28), Third Molar (38), Third Molar (48)
```

## Results

The notebook logs mAP, precision, recall, and class-wise statistics during training. You can visualize predictions and further analyze results for reporting and research.

## License

This repository is provided for academic research and educational purposes. Please consult LICENSE for terms of use.

## References

- [Ultralytics YOLO Documentation](https://docs.ultralytics.com/)
- [FDI World Dental Federation Notation](https://en.wikipedia.org/wiki/FDI_World_Dental_Federation_notation)

---

**Quick Start:**  
Run `OravisHealthCare_Updated.ipynb` in Google Colab or locally following the steps above!
