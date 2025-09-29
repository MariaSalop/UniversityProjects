# Computer Vision – Root Length Detection  
**Author:** Maria Salop – Breda University of Applied Sciences  

---

## Overview  
This project applies **computer vision** and **deep learning (U-Net)** to measure **primary root lengths** of plants grown in Petri dishes. The workflow combines image preprocessing, CNN-based segmentation, and skeleton analysis to automatically detect and quantify root lengths.  

The work was part of an academic project in Applied Data Science & AI.  

---

## Objectives  
- Develop a reproducible pipeline for **root segmentation** using U-Net.  
- Post-process masks to isolate **primary roots** from background noise and seeds.  
- Divide Petri dish images into **five equal plant regions** for individual measurement.  
- Compute **root length** in pixels using skeleton shortest path analysis.  
- Export final results to CSV for further analysis.  

---

## Repository Structure  

```
ComputerVision_RootLength/
├─ README.md
├─ RootLength_Detection.ipynb                # end-to-end pipeline
├─ src/
│  └─ task_8_helpers.py                      # f1, crop_to_petri_dish, padder, RSA utils
├─ models/
│  └─ SalopMaria_230574_unet_model_128px.h5  # trained U-Net weights
├─ data/
│  ├─ images/                                # input plates (not pushed to repo)
│  ├─ masks/                                 # model outputs (18)
│  └─ divided_masks/                         # 5 per plate (90)
├─ results/
│  └─ primary_root_lengths_v2.csv            # per-plant primary root lengths (px)
└─ presentation/
    └─ RootLength_Project_Presentation.pdf    # slides
```

---

## Methodology  

1. **Preprocessing**  
   - Crop Petri dish regions.  
   - Pad images for patch-based prediction.  

2. **Segmentation**  
   - Apply U-Net (patch size 128×128, stride 128).  
   - Threshold prediction (0.25) to create binary masks.  

3. **Plant Separation**  
   - Divide each Petri dish mask into **five vertical sections**, one per plant.  

4. **Root Isolation**  
   - Remove small/noisy components (seed regions).  
   - Retain the **largest vertically spanning root** as the primary root.  

5. **Length Calculation**  
   - Skeletonize the binary mask.  
   - Compute **shortest path length** from the root tip (top) to bottom point.  
   - Save results to `primary_root_lengths_v2.csv`.  

---

## Results  

- **18 plates processed**, each with **5 plants** → total of **90 root lengths** measured.  
- Outputs stored in:  
  - `results/primary_root_lengths_v2.csv` → root lengths in pixels.  
  - Example entry:  
    ```
    Plant ID,Length (px)
    test_image_17_plant_4,1029
    ```
- Values are in **pixels**. Conversion to millimeters requires pixel-to-mm calibration (not included).  

---

## Presentation  

The folder `presentation/` contains:  

- `RootLength_Project_Presentation.pdf` – slides summarizing the project, including:  
  - Context and motivation.  
  - U-Net segmentation workflow.  
  - Example masks and plant divisions.  
  - Results overview.  

This provides a concise visual overview for quick understanding.  

