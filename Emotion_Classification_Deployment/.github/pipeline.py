import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import contractions
import pandas as pd
import requests
import tensorflow as tf
import torch
import whisper
from datasets import Dataset
from emoji import demojize
from pytubefix import YouTube
from textblob import TextBlob
from tqdm import tqdm
from transformers import (AutoModelForSequenceClassification,
                          AutoTokenizer, TFAutoModelForSeq2SeqLM, Trainer)

API_BASE = "http://194.171.191.227:30080"
token = "sk-957c1a6611e44e619c893b01d45d08b0"
header = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

SLANG_DICT = {
    # Common abbreviations
    "lol": "laughing out loud",
    "omg": "oh my god",
    "btw": "by the way",
    "idk": "i don't know",
    "tbh": "to be honest",
    "imo": "in my opinion",
    "smh": "shaking my head",
    "afaik": "as far as i know",
    "fyi": "for your information",
    "np": "no problem",
    "thx": "thanks",
    "pls": "please",
    "asap": "as soon as possible",
    "jk": "just kidding",
    "nvm": "never mind",
    "brb": "be right back",
    "gtg": "got to go",
    "irl": "in real life",
    "dm": "direct message",
    "tmi": "too much information",
    # Emphatic expressions
    "wtf": "what the fuck",
    "omfg": "oh my fucking god",
    "stfu": "shut the fuck up",
    "fml": "fuck my life",
    "rofl": "rolling on the floor laughing",
    "lmao": "laughing my ass off",
    "lmfao": "laughing my fucking ass off",
    # Modern internet slang
    "sus": "suspicious",
    "ghosting": "ignoring someone",
    "simp": "someone idolizing others",
    "flex": "showing off",
    "clout": "influence",
    "vibe": "mood",
    "yeet": "throw forcefully",
    "lit": "exciting",
    "salty": "bitter/angry",
    "cap": "lie",
    "no cap": "truth",
    "bet": "agreement",
    "ship": "relationship",
    "stan": "obsessed fan",
    # Textspeak conversions
    "u": "you",
    "ur": "your",
    "r": "are",
    "y": "why",
    "k": "okay",
    "ppl": "people",
    "def": "definitely",
    "prob": "probably",
    "gonna": "going to",
    "wanna": "want to",
    "gotta": "got to",
}
SLANG_DICT.update({
    "gg": "good game",
    "op": "overpowered",
    "nerf": "reduce power",
    "pog": "awesome"
})


def optimized_preprocessor(text):
    """
    Minimal yet effective preprocessing for transformer models
    Returns: Cleaned text string
    """
    # Convert emojis to text descriptions
    text = demojize(text, delimiters=(" ", " "))
    # 👌🏾 → :ok_hand_medium-dark_skin_tone:

    # Expand slang/abbreviations
    text = " ".join([
        SLANG_DICT.get(word.lower(), word)
        for word in text.split()
    ])

    # Handle repeated characters (e.g., "loooool" → "lool")
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)

    # Remove remaining special characters
    # (keep apostrophes and basic punctuation)
    text = re.sub(r"[^a-zA-Z0-9\s!?,;:'\-]", "", text)

    # Handle contractions (e.g., "can't" → "cannot")
    text = contractions.fix(text)

    # Remove user mentions (@username)
    text = re.sub(r"@\w+", "[USER]", text)

    # Remove URLs
    text = re.sub(r"http\S+", "[URL]", text)

    # Normalize numbers
    text = re.sub(r"\d+", "[NUM]", text)

    # Convert to lowercase
    text = text.lower()

    return text


def remove_placeholders(text):
    """
    Removes placeholders [NUM], [USER], and [URL] from the text.
    """
    # Remove [NUM], [USER], and [URL]
    text = re.sub(r"\[NUM\]", "", text)
    text = re.sub(r"\[USER\]", "", text)
    text = re.sub(r"\[URL\]", "", text)

    # Optionally, strip any extra spaces that might be left after removal
    text = " ".join(text.split())

    return text


def transcribe_audio(mp3_file: str, model_name: str = "base") -> str:
    """
    Transcribes the given MP3 file using the specified Whisper model.

    Args:
        mp3_file (str): Path to the MP3 file.
        model_name (str): Name of the Whisper model to use
                            (tiny, base, small, medium, large).

    Returns:
        str: The full transcription text.
    """
    print(f"Loading Whisper model '{model_name}'...")
    model = whisper.load_model(model_name)
    print(f"Transcribing '{mp3_file}'...")
    result = model.transcribe(mp3_file)
    return result["text"]


def split_into_sentences(text: str) -> list:
    """
    Splits a block of text into sentences using a regular expression.
    This regex splits the text at punctuation marks (., !, or ?)
    followed by whitespace.

    Args:
        text (str): The text to split.

    Returns:
        list: A list of sentences.
    """
    # The regex splits on punctuation that likely ends a sentence.
    sentences = re.split(r"(?<=[.!?])\s+", text)
    # Clean up any extra whitespace or empty strings.
    sentences = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]
    return sentences


def save_sentences_to_csv(sentences: list, output_file: str) -> None:
    """
    Saves a list of sentences into a CSV file with one column "Sentence".
    Uses UTF-8 encoding with BOM to properly display Romanian characters.

    Args:
        sentences (list): List of sentence strings.
        output_file (str): Path to the output CSV file.
    """
    df = pd.DataFrame(sentences, columns=["Sentence"])
    df.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig"
    )
    print(f"Transcription saved to '{output_file}'.")


def download_youtube_audio(url, destination="video_to_mp3"):
    try:
        yt = YouTube(url)
        title = yt.title
        sanitized_title = "".join(
            c for c in title if c.isalnum() or c in " _-"
        ).rstrip()
        mp3_filename = os.path.join(destination, sanitized_title + ".mp3")

        # Check if the file already exists
        if os.path.exists(mp3_filename):
            print(f"File already downloaded: {mp3_filename}")
            return mp3_filename

        # If not, download it
        video = yt.streams.filter(only_audio=True).first()
        out_file = video.download(output_path=destination)
        base, ext = os.path.splitext(out_file)
        new_file = base + ".mp3"
        if os.path.exists(new_file):
            print(f"File already converted: {new_file}")
            return new_file

        os.rename(out_file, new_file)

        print(f"Download complete: {new_file}")
        return new_file

    except Exception as e:
        print("Failed to download audio:", e)
        return None


def handle_mp3_file(path):
    if os.path.isfile(path) and path.lower().endswith(".mp3"):
        print(f"MP3 file found at: {path}")
        return path
    else:
        print("Invalid file path or not an MP3.")
        return None


# Translation function using trained model
def translate_text(text, tokenizer, model):
    tokenized = tokenizer(
        [text],
        return_tensors="np",
        padding=True,
        truncation=True,
        max_length=128
    )
    out = model.generate(**tokenized, max_length=128)
    return tokenizer.decode(out[0], skip_special_tokens=True)


# Function to calculate the absolute polarity
def get_abs_polarity(text):
    blob = TextBlob(text)
    return abs(blob.sentiment.polarity)


# Function to categorize intensity based on absolute polarity
def categorize_intensity(abs_polarity):
    if abs_polarity >= 0.9:
        return "extremely intense"
    elif abs_polarity >= 0.7:
        return "very intense"
    elif abs_polarity >= 0.5:
        return "intense"
    elif abs_polarity >= 0.3:
        return "moderate"
    elif abs_polarity >= 0.1:
        return "mild"
    elif abs_polarity >= 0.05:
        return "slightly mild"
    else:
        return "low"


# Helper function for API call
def correct_transcription_romanian(text, system_prompt, api_url, headers):
    payload = {
        "model": "llama3.3:latest",  # Example: Adjust model if needed
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        "options": {"num_ctx": 4096 * 2},
        "temperature": 0.5,
        "top_p": 0.2,
    }

    try:
        response = requests.post(
            f"{api_url}/api/chat/completions",
            headers=headers,
            json=payload,
            timeout=180,
        )
        result = response.json()
        choices = result.get("choices", [])
        if choices and "message" in choices[0]:
            return choices[0]["message"]["content"].strip()
        else:
            return "error"
    except Exception:
        return "error"


# Process sentences and correct transcriptions
def correct_sentences(sentences, system_prompt, api_url, headers):
    corrected_sentences = [None] * len(
        sentences
    )  # List to store corrected sentences in order

    with ThreadPoolExecutor(max_workers=200) as executor:
        futures = {
            executor.submit(
                correct_transcription_romanian,
                sentence,
                system_prompt,
                api_url,
                headers,
            ): idx
            for idx, sentence in enumerate(sentences)
        }

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="🔍 Correcting transcriptions with LLaMA",
        ):
            idx = futures[future]
            try:
                corrected = future.result()
                if corrected != "error":
                    original_words = sentences[idx].strip().split()
                    corrected_words = corrected.strip().split()

                    # If difference in word count is small enough, accept it
                    if abs(len(corrected_words) - len(original_words)) <= 2:
                        corrected_sentences[idx] = corrected
                    else:
                        # Keep original
                        corrected_sentences[idx] = sentences[idx]
                else:
                    # On error, keep original
                    corrected_sentences[idx] = sentences[idx]
            except Exception:
                corrected_sentences[idx] = sentences[
                    idx
                ]  # Keep original sentence in case of error

    return corrected_sentences


def main():
    prompt = "Enter a YouTube link or path to an .mp3 file:\n"
    user_input = input(prompt).strip()
    if user_input.startswith("http://") or user_input.startswith("https://"):
        file_path = download_youtube_audio(user_input)
    else:
        file_path = handle_mp3_file(user_input)

    if not file_path or not os.path.isfile(file_path):
        print(f"Error: The file '{file_path}' does not exist.")
        exit(1)
    print(f"File path: {file_path}")
    # Transcribe the audio file.
    transcription_text = transcribe_audio(file_path, model_name="large")

    # Split the transcription into sentences.

    sentences = split_into_sentences(transcription_text)
    # Romanian Transcription Correction Prompt
    system_prompt_romanian = """
    You are an assistant that helps correct transcription errors in Romanian.
    Below is a sentence.
    If there are any mistakes, correct only the spelling mistakes
    — do not replace words with synonyms or change their meaning.
    Keep the original words unless there is
    a clear spelling or transcription error.
    Do not rephrase or rewrite. Only output the corrected sentence.

    Sentence:
    """

    # Correct the sentences
    corrected_sentences = correct_sentences(
        sentences, system_prompt_romanian, API_BASE, header
    )

    # Determine output file name
    output_file = "audio_transcript.csv"

    # Save the sentences to CSV.
    save_sentences_to_csv(sentences, output_file)
    # Reload trained model for translation
    tokenizer_translate = AutoTokenizer.from_pretrained(
        "models/models/translation/"
    )
    # Force the model to run on CPU
    with tf.device("/cpu:0"):
        model_translate = TFAutoModelForSeq2SeqLM.from_pretrained(
            "models/models/translation/"
        )
    df = pd.read_csv("audio_transcript.csv")
    df["Corrected_Sentence"] = corrected_sentences
    # Apply translation using trained model
    df["Translation"] = (
        df["Corrected_Sentence"]
        .astype(str)
        .apply(
            lambda x: translate_text(
                x, tokenizer=tokenizer_translate, model=model_translate
            )
        )
    )

    # Save the result as an Excel file
    df.to_csv("translated_data.csv", index=False, encoding="utf-8-sig")
    print("Translated sentences saved to 'translated_data.csv'.")

    model_path = r"models/models/emotion_7"

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    # Move model to device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    df["Preprocessed_Translation"] = df["Translation"].apply(
        optimized_preprocessor
    )
    df["Preprocessed_Translation"] = df["Preprocessed_Translation"].apply(
        remove_placeholders
    )
    # Load or define df_test here
    dataset = Dataset.from_pandas(df[["Preprocessed_Translation"]])

    # Tokenize
    def tokenize_function(example):
        return tokenizer(
            example["Preprocessed_Translation"],
            padding="max_length",
            truncation=True,
            max_length=128,
        )

    tokenized_dataset_test = dataset.map(tokenize_function, batched=True)
    tokenized_dataset_test = tokenized_dataset_test.remove_columns(
        ["Preprocessed_Translation"]
    )
    tokenized_dataset_test.set_format(
        type="torch", columns=["input_ids", "attention_mask"]
    )
    # Recreate the trainer
    trainer = Trainer(model=model, tokenizer=tokenizer)
    predictions_output = trainer.predict(tokenized_dataset_test)

    # Extract logits and labels
    logits = predictions_output.predictions

    # Convert logits to predicted label indices
    predicted_labels = logits.argmax(axis=1)

    # Define label mappings:
    # happiness - sadness - anger - surprise - fear - disgust
    label_mapping = {
        "disgust": 0,
        "happiness": 1,
        "anger": 2,
        "neutral": 3,
        "sadness": 4,
        "fear": 5,
        "surprise": 6,
    }
    # Reverse mapping to decode label integers
    reverse_label_mapping = {v: k for k, v in label_mapping.items()}

    df["Core_Emotion"] = predicted_labels
    df["Core_Emotion"] = df["Core_Emotion"].map(reverse_label_mapping)

    label_mapping_fine = {
        "accepted": 0,
        "angry": 1,
        "confused": 2,
        "excited": 3,
        "fearful": 4,
        "frustrated": 5,
        "grossed out": 6,
        "guilty": 7,
        "humiliated": 8,
        "hurt": 9,
        "interested": 10,
        "joyful": 11,
        "nervous": 12,
        "neutral": 13,
        "optimistic": 14,
        "outraged": 15,
        "peaceful": 16,
        "proud": 17,
        "startled": 18,
    }

    model_path = r"models/models/emotion_19"
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    tokenized_dataset_test = dataset.map(tokenize_function, batched=True)
    tokenized_dataset_test = tokenized_dataset_test.remove_columns(
        ["Preprocessed_Translation"]
    )
    tokenized_dataset_test.set_format(
        type="torch", columns=["input_ids", "attention_mask"]
    )
    # Recreate the trainer
    trainer = Trainer(model=model, tokenizer=tokenizer)
    predictions_output = trainer.predict(tokenized_dataset_test)

    # Extract logits and labels
    logits = predictions_output.predictions

    # Convert logits to predicted label indices
    predicted_labels = logits.argmax(axis=1)
    # Reverse mapping to decode label integers
    reverse_label_mapping = {v: k for k, v in label_mapping_fine.items()}
    df["Fine_Emotion"] = predicted_labels
    df["Fine_Emotion"] = df["Fine_Emotion"].map(reverse_label_mapping)

    # Apply the function to the sentences column
    df["Intensity"] = df["Preprocessed_Translation"].apply(get_abs_polarity)
    # Categorize the intensity based on absolute polarity
    df["Intensity"] = df["Intensity"].apply(categorize_intensity)
    df.to_csv("pipeline_output_3.csv", index=False, encoding="utf-8-sig")
    print("Pipeline output saved to 'pipeline_output_3.csv'.")
    print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
