import pandas as pd
from pathlib import Path


def load_and_sample_dataset(
    input_path: str,
    output_path: str,
    n: int = 200
) -> None:
    """Load full dataset and write a stratified sample to CSV."""
    df = pd.read_csv(input_path)
    sampled = (
        df.groupby("predicted_emotion")
        .apply(lambda x: x.head(n))
        .reset_index(drop=True)
    )
    sampled.to_csv(output_path, index=False)


def create_readme(path: str) -> None:
    """Create a detailed README describing the sample dataset."""
    readme = """\
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
"""
    Path(path).write_text(readme.strip())


def main() -> None:
    """Create sample transcript and dataset README."""
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    load_and_sample_dataset(
        input_path="core_dataset_v2.csv",
        output_path=data_dir / "sample_transcript.csv"
    )

    create_readme(path=data_dir / "README.md")


if __name__ == "__main__":
    main()
