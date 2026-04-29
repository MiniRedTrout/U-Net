# Aortic Segmentation with U-Net

##  Overview
This project presents a deep learning solution for automatic **aortic segmentation** from medical images (e.g., CT or MRI). A **U-Net** architecture is implemented to achieve pixel‑wise segmentation of the aorta. The model attains a **Dice similarity coefficient of 98%**, demonstrating high accuracy and clinical applicability.

Additionally, a **3D anatomical model** of the aorta is reconstructed from the segmented 2D slices, enabling enhanced visualization, morphological analysis, and potential pre‑surgical planning.

##  Key Features
- **High accuracy**: 98% Dice score on test data.
- **Robust U‑Net architecture**: Adapted for 2D medical image segmentation.
- **3D reconstruction**: Volumetric visualization from sequential 2D masks.
- **Modular codebase**: Easy to train, evaluate, and apply to new data.

##  Results
| Metric        | Score  |
|---------------|--------|
| Dice Score    | 98%    |
| IoU (Jaccard) | ~96.2% |
| Pixel Accuracy| ~99.1% |
