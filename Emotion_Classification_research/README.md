# Emotion Classification – Research Project
**Author:** Maria Salop – Breda University of Applied Sciences  

---

## Overview
This repository contains the deliverables for the **Emotion Classification (Research Project)**, developed as part of the Applied Data Science & AI program.  
The focus was on building an **end-to-end NLP pipeline** to classify emotions from text and speech, with tasks covering **annotation, speech-to-text, machine translation, traditional & deep learning models, explainability (XAI)**, and **error analysis**.  

Unlike the deployment-focused project, this repository emphasizes **research methods**, model exploration, and academic reporting.

---

## Objectives
- Annotate and prepare emotion-labeled data for training.  
- Implement speech-to-text models and evaluate them with **Word Error Rate (WER)**.  
- Perform **feature extraction** and create a **data quality report**.  
- Train and compare multiple models (traditional ML + deep learning).  
- Develop a custom **machine translation model** to expand the dataset.  
- Use **prompt engineering** for automated labeling.  
- Apply **Explainable AI (XAI)** techniques to transformer models.  
- Conduct a structured **error analysis** of model predictions.  
- Produce a **model card** summarizing the data, models, and ethical considerations.  
- Deliver a final presentation of the findings.  

---

## Repository Structure

```text
emotion-classification-research/
├─ README.md
├─ notebooks/
│  ├─ feature_extraction.ipynb       # extracting NLP features
│  ├─ model_iterations.ipynb         # training traditional + DL models
│  ├─ pipeline.ipynb                 # end-to-end pipeline assembly
│  ├─ prompt_engineering.ipynb       # prompt-based labeling
│  ├─ STT_Assembly.ipynb             # speech-to-text with AssemblyAI
│  ├─ STT_Whisper.ipynb              # speech-to-text with Whisper
│  └─ transcribed_data_assembly_wer.xlsx  # WER evaluation (Assembly)
│  └─ transcribed_data_whisper_wer.xlsx   # WER evaluation (Whisper)     # explainability methods
├─ reports/
│  ├─ error_analysis.pdf
│  ├─ XAI_Roberta.pdf
│  ├─ model_card.md
│  └─ final_presentation.pdf
└─ README.md

```