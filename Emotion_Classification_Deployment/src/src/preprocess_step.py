# src/preprocess_step.py
"""CLI entry-point used by the Azure ML pipeline to clean raw text data.

The script:
1. reads a CSV file containing a column with raw text;
2. applies the project's `optimized_preprocessor` to every row;
3. writes a new CSV that contains the cleaned text.

Example (local):
python preprocess_step.py \
       --input_csv data/raw.csv \
       --output_csv data/clean/clean.csv \
       --text_col text \
       --clean_col clean_text
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# Re-use the existing team helper — nothing heavy, already covered by unit tests
from project_nlp8.textclean import optimized_preprocessor


def clean_dataframe(df: pd.DataFrame, *, text_col: str, clean_col: str) -> pd.DataFrame:
    """Return a copy of *df* with an extra column that holds the cleaned text."""
    df_out = df.copy()
    df_out[clean_col] = df_out[text_col].astype(str).apply(optimized_preprocessor)
    return df_out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-process raw dataset for emotion classification"
    )
    parser.add_argument(
        "--input_csv",
        type=Path,
        required=True,
        help="Path to the raw CSV file"
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        required=True,
        help="Destination for the cleaned CSV"
    )
    parser.add_argument(
        "--text_col",
        type=str,
        default="text",
        help="Name of the raw-text column"
    )
    parser.add_argument(
        "--clean_col",
        type=str,
        default="clean_text",
        help="Name of the new cleaned column"
    )
    args = parser.parse_args()

    df_raw = pd.read_csv(args.input_csv)
    df_clean = clean_dataframe(df_raw, text_col=args.text_col, clean_col=args.clean_col)

    # Keep Azure ML happy – make sure directory exists
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    main()
