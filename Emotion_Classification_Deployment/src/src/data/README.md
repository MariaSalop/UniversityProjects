# Survivor-RO Sample Dataset

This dataset is a reduced and labeled subset of the original 
Survivor-RO dialogue dataset. It is intended for internal use during 
the development of NLP pipelines that include emotion classification 
tasks.

## Source and Licensing

**Source:** Internal dataset: `core_dataset_v2.csv`  
**License:** Research only – derived from Survivor-RO dataset  
**Usage Terms:**  
- Not for commercial use  
- Redistribution not permitted  
- Attribution required for external use  

## Dataset Overview

**Sample File:**  
- `data/sample_transcript.csv`  
- Contains 200 randomly selected utterances per emotion category  
- Extracted from the full dataset (`core_dataset_v2.csv`)

**Emotion Classes (7 total):**  
- anger  
- joy  
- optimism  
- sadness  
- surprise  
- fear  
- disgust  

### Structure

| Column             | Description                           |
|--------------------|---------------------------------------|
| `text`             | Spoken sentence or utterance          |
| `predicted_emotion`| Associated emotion label (one of 7)   |

## Intended Usage

This sample dataset is designed for:  
- Prototyping ML pipelines  
- Initial emotion detection experiments  
- Unit testing and evaluation of modeling infrastructure

## Notes

- Total rows: 1400 (200 × 7 emotions)  
- File format: CSV (UTF-8 encoding, no index column)  
- Saved at: `data/sample_transcript.csv`