# Emotion Classification Deployment

## Overview
This project extends our earlier emotion classification research into a full **deployable pipeline**.  
It combines speech-to-text, preprocessing, translation, and a transformer-based emotion classifier into an **end-to-end product** with local inference, containerization, and cloud deployment.  

The system supports:
- **Speech-to-text transcription**
- **Optional text preprocessing** (emoji demojizing, slang expansion, placeholder removal)
- **Translation**
- **Emotion classification** using a fine-tuned RoBERTa model
- **Model deployment via FastAPI, Docker, and Azure ML**
- **Streamlit demo frontend**

---

## Objectives
- Build an installable Python package for the complete pipeline.
- Provide a REST API for inference with FastAPI.
- Containerize the training and inference code with Docker.
- Deploy the solution to **Microsoft Azure ML** for scalability.
- Provide a **Streamlit demo** to showcase the deployed model.
- Document project backlogs, architecture, and roadmap. 

---

## Methodology
1. **Data Collection & Annotation**  
   Emotion dataset curated and annotated for classification.  

2. **Feature Engineering & Preprocessing**  
   - Tokenization  
   - Emoji → text conversion  
   - Slang expansion  

3. **Modeling**  
   - Fine-tuned **RoBERTa** model using Hugging Face Transformers.  
   - Evaluation with accuracy, F1-score, and confusion matrices.  

4. **Deployment**  
   - FastAPI for REST inference.  
   - Docker container for portability.  
   - Azure ML for cloud deployment.  
   - Streamlit for interactive demo.  
