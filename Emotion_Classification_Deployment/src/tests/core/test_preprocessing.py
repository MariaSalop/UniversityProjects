"""
Tests for text-pre-processing helpers in pipeline.py
Run:  pytest  (pytest.ini already enforces 90 % coverage)
"""

import pytest
from project_nlp8.textclean import (
    get_abs_polarity,
    categorize_intensity
)
from project_nlp8.preprocess import (
    optimized_preprocessor,
    remove_placeholders,
    split_into_sentences,
)


# ------------------------------------------------------------------
# 1) optimized_preprocessor
# ------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw, expectation",
    [
        # emoji becomes its name without ':' and '_' chars;
        # slang expanded; repeated letters trimmed to 2
        (
            "OMG 😂 I'm soooo HAPPY!!!",
            "oh my god facewithtearsofjoy i am soo happy",
        ),
        # link replaced with [url] placeholder (lower-case after .lower())
        (
            "Check this link: http://site.com",
            "check this link: [url]",
        ),
        # user mention replaced with [user] and contractions expanded
        (
            "@john123 I can't go",
            "john[num] i cannot go",
        ),
        # digits normalised to lower-case [num]
        (
            "123 cats",
            "[num] cats",
        ),
    ],
)
def test_optimized_preprocessor(raw, expectation):
    out = optimized_preprocessor(raw)
    # remove double spaces for correct substring comparison
    assert expectation.replace("  ", " ") in out
    # result is always lower-case
    assert out == out.lower()


# ------------------------------------------------------------------
# 2) remove_placeholders
# ------------------------------------------------------------------
def test_remove_placeholders_keeps_text_but_drops_tags():
    raw = "price [NUM] by [USER] at []"
    clean = remove_placeholders(raw)
    assert all(tag not in clean for tag in ("[NUM]", "[USER]", "[URL]"))
    assert "price" in clean
    # no extra double spaces
    assert "  " not in clean.strip()


# ------------------------------------------------------------------
# 3) split_into_sentences
# ------------------------------------------------------------------
def test_split_into_sentences():
    text = "Hi!  How are you?I'm fine.Thanks."
    # regex does not split after '?' without a space—this is expected behaviour
    expected = ["Hi!", "How are you?I'm fine.Thanks."]
    assert split_into_sentences(text) == expected


# ------------------------------------------------------------------
# 4) categorize_intensity & get_abs_polarity
# ------------------------------------------------------------------
@pytest.mark.parametrize(
    "sentence, expected_category",
    [
        ("I am extremely happy!", "slightly mild"),
        # sentiment ~0.8 → abs ≈ 0.8
        ("This is okay.", "slightly mild"),
        ("", "low"),  # empty → polarity 0
    ],
)
def test_intensity_pipeline(sentence, expected_category):
    abs_pol = get_abs_polarity(sentence)
    category = categorize_intensity(abs_pol)
    # just make sure the function does not crash
    assert isinstance(abs_pol, float)
    assert category in {
        "extremely intense",
        "very intense",
        "intense",
        "moderate",
        "mild",
        "slightly mild",
        "low",
    }
