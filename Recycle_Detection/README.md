# Recycle Detection – CNN Model  
*Maria Salop – Breda University of Applied Sciences*  

## Overview  
This project, titled **“The Future of Recycling”**, applied **Deep Learning (CNNs)** to classify recyclable waste materials into four categories: **paper, plastic, metal, and organic**.  
It demonstrates how AI can contribute to sustainability, smart waste management, and improved recycling efficiency.  

## Problem Statement  
- Inefficient waste management leads to high costs and environmental damage.  
- Recycling practices often lack accuracy and scalability.  
- **Goal:** Build a computer vision model that can **automatically classify waste items** to support recycling efficiency and sustainability.  

## Dataset  
- **Paper**: 182 images  
- **Plastic**: 141 images  
- **Metal**: 164 images  
- **Organic**: 132 images  
- Baselines:  
  - Random guess: 25%  
  - Human-level: 100%  
  - Basic MLP: 32%  

## Model Development  
The project followed multiple iterations to improve performance:  
1. **CNN (3 conv layers + normalization)** – Training Accuracy: 90%  
2. **CNN + Batch Norm + Dense** – Training Accuracy: 90%  
3. **CNN + Data Augmentation (ImageDataGenerator)** – Training Accuracy: 99%  
4. **Transfer Learning with VGG16** – Training Accuracy: 59%  

**Final Model Architecture**  
- Base: VGG16 (frozen pretrained layers on ImageNet)  
- Custom Layers: Flatten → Dense(256, ReLU) → Dense(128, ReLU) → Dropout(50%) → Dense(4, Softmax)  
- Optimizer: Adam with Sparse Categorical Crossentropy loss  

## Results  
- **Final Training Accuracy:** 99.7%  
- **Validation Accuracy:** 69.7%  
- **Confusion Matrix:** Model achieved strong recognition of paper & metal, with weaker generalization for plastic and organic.  

| Iteration | Accuracy | Precision | Recall | F1 Score |
|-----------|----------|-----------|--------|----------|
| Iteration 1 | 77% | 77% | 76% | 76% |
| Iteration 2 | 31% | 19% | 31% | 23% |
| Iteration 3 | 68% | 69% | 69% | 69% |
| Iteration 4 | 67% | 55% | 61% | 56% |

## Responsible AI  
- **High Accuracy Priority:** Ensures correct classification and boosts recycling efficiency.  
- **Interpretability:** Helps educate users on waste segregation, improves trust and app adoption.  
- **Balance:** Model design prioritizes both accuracy and interpretability.  

## Human-Centered AI – User Study  
- Conducted a **UX study** on app navigation and search functionality.  
- Hypotheses:  
  1. Simplified navigation reduces frustration and increases speed.  
  2. Enhanced search improves satisfaction and result relevance.  
- Incorporated results into wireframe prototype with **auto-suggestions, filters, and grouped menu items**.  

## Deliverable  
- A **deep learning recycling classifier** that categorizes waste into 4 classes.  
- A **prototype design** demonstrating how AI-based recycling apps could enhance **sustainability and user engagement**.  

---
