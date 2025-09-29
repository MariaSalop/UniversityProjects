[![codecov](https://codecov.io/gh/BredaUniversityADSAI/2024-25d-fai2-adsai-group-nlp8/graph/badge.svg?token=AFAMFE8ON6)](https://codecov.io/gh/BredaUniversityADSAI/2024-25d-fai2-adsai-group-nlp8)

[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/uhJ9rk0N)

# Audio Transcription and Sentiment Analysis Pipeline

This pipeline transcribes audio from MP3 files or YouTube videos, splits the transcription into sentences, corrects Romanian transcriptions, translates sentences, and performs sentiment analysis. It leverages several machine learning models and APIs to process the data.

## Requirements

- Python 3.8+
- `requirements.txt` contains all the dependencies for the project.

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/BredaUniversityADSAI/2024-25c-fai2-adsai-group-group_26_y2c.git
    ```
    ```bash
   cd https://github.com/BredaUniversityADSAI/2024-25c-fai2-adsai-group-group_26_y2c.git
   ```

2. Create a virtual environment:

    ```bash
    python3 -m venv venv
    ```

3. Activate the virtual environment:

- On Windows:
    ```bash
    venv\Scripts\activate
    ```
- On Mac/Linux:
    ```bash
    source venv/bin/activate
    ```

4. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Make sure you have the necessary models and datasets downloaded and saved in the appropriate directories *(models/models/translation/*, *models/models/emotion_7/*).

2. Run the script using:

    ```bash
    python pipeline.py
    ```

3. When prompted, enter a YouTube link or the path to an MP3 file for transcription. The script will handle downloading, transcribing, translating, and performing sentiment analysis.

4. The final output will be saved in two CSV files:

    - audio_transcript.csv: The original transcription split into sentences.

    - translated_data.csv: The translated and sentiment-analyzed data.


## Script Overview

The main steps of the pipeline are as follows:

1. Download YouTube Audio: If the input is a YouTube link, the audio is downloaded and converted to an MP3 file.

2. Audio Transcription: The MP3 file is transcribed using the Whisper model.

3. Text Preprocessing: The transcribed text is split into sentences and corrected using a Romanian transcription model.

4. Translation: The corrected sentences are translated using a translation model.

5. Sentiment Analysis: The translated sentences are preprocessed and analyzed for emotions using a trained sentiment model.

6. Save Results: The final results are saved as CSV files.
