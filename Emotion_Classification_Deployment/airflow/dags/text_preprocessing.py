import re
import pandas as pd
import contractions
from emoji import demojize
 
# Dictionary to expand slang
SLANG_DICT = {
    "lol": "laughing out loud", "omg": "oh my god", "btw": "by the way", "idk": "i don't know",
    "tbh": "to be honest", "imo": "in my opinion", "smh": "shaking my head",
    "afaik": "as far as i know", "fyi": "for your information", "np": "no problem", "thx": "thanks",
    "pls": "please", "asap": "as soon as possible", "jk": "just kidding", "nvm": "never mind",
    "brb": "be right back", "gtg": "got to go", "irl": "in real life", "dm": "direct message",
    "tmi": "too much information", "wtf": "what the fuck", "omfg": "oh my fucking god",
    "stfu": "shut the fuck up", "fml": "fuck my life", "rofl": "rolling on the floor laughing",
    "lmao": "laughing my ass off", "lmfao": "laughing my fucking ass off", "sus": "suspicious",
    "ghosting": "ignoring someone", "simp": "someone idolizing others", "flex": "showing off",
    "clout": "influence", "vibe": "mood", "yeet": "throw forcefully", "lit": "exciting",
    "salty": "bitter/angry", "cap": "lie", "no cap": "truth", "bet": "agreement", "ship": "relationship",
    "stan": "obsessed fan", "u": "you", "ur": "your", "r": "are", "y": "why", "k": "okay",
    "ppl": "people", "def": "definitely", "prob": "probably", "gonna": "going to",
    "wanna": "want to", "gotta": "got to"
}
 
def optimized_preprocessor(text: str) -> str:
    """
    Cleans and standardizes input text by performing the following:
    - Replaces emojis with their textual descriptions.
    - Expands slang based on SLANG_DICT.
    - Reduces repeated characters (more than two).
    - Removes non-standard characters except common punctuation.
    - Expands contractions.
    - Masks usernames, URLs, and numeric strings.
    - Converts to lowercase.
 
    Args:
        text (str): Input text string.
 
    Returns:
        str: Preprocessed and cleaned text.
    """
    if not isinstance(text, str):
        return ""
    text = demojize(text, delimiters=(" ", " "))
    text = ' '.join([SLANG_DICT.get(word.lower(), word) for word in text.split()])
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = re.sub(r"[^a-zA-Z0-9\s!?,;:'\-]", "", text)
    text = contractions.fix(text)
    text = re.sub(r"@\w+", "[USER]", text)
    text = re.sub(r"http\S+", "[URL]", text)
    text = re.sub(r"\d+", "[NUM]", text)
    return text.lower()
 
def remove_placeholders(text: str) -> str:
    """
    Removes placeholder tokens ([NUM], [USER], [URL]) from the text.
 
    Args:
        text (str): Text containing placeholder tokens.
 
    Returns:
        str: Text with placeholders removed and cleaned of extra spaces.
    """
    text = re.sub(r"\[NUM\]|\[USER\]|\[URL\]", "", text)
    return ' '.join(text.split())
 
def preprocess_pipeline(text: str, keep_placeholders=True):
    """
    Runs the full preprocessing pipeline on input text.
 
    Args:
        text (str): Input text string.
        keep_placeholders (bool): Whether to keep placeholder tokens.
 
    Returns:
        str: Fully preprocessed text.
    """
    text = optimized_preprocessor(text)
    if not keep_placeholders:
        text = remove_placeholders(text)
    return text
 
def preprocess_csv(input_path: str, output_path: str, text_column: str = "text", keep_placeholders=True):
    """
    Loads a CSV file, applies text preprocessing to a specified column, 
    and saves the result to a new CSV with an additional 'processed_text' column.
 
    Args:
        input_path (str): Path to the input CSV file.
        output_path (str): Path to save the processed CSV file.
        text_column (str): Name of the column containing the text to preprocess.
        keep_placeholders (bool): Whether to retain placeholder tokens.
 
    Raises:
        ValueError: If the specified text_column is not found in the input CSV.
    """
    df = pd.read_csv(input_path)
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in CSV.")
    df["processed_text"] = df[text_column].apply(lambda x: preprocess_pipeline(x, keep_placeholders))
    df.to_csv(output_path, index=False)