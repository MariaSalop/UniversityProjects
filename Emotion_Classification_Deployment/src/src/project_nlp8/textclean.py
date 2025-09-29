import re
import contractions
from emoji import demojize
from textblob import TextBlob


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
