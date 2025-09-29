"""
Logic for reading a CSV of texts, predicting on each row,
and writing out a new CSV as a string buffer.
"""

import csv
import io
import pandas as pd
from .predictor import predict_text


def process_text_csv(file_stream) -> str:
    """
    Read CSV from file_stream (e.g. UploadFile.file),
    expect a 'text' column,
    return new CSV as a string with columns text,predicted_label.
    """
    df = pd.read_csv(file_stream)
    if "text" not in df.columns:
        raise ValueError("CSV must contain a 'text' column")

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["text", "predicted_label"])

    for txt in df["text"].astype(str):
        txt = txt.strip()
        if not txt:
            continue
        pred = predict_text(txt)
        writer.writerow([txt, pred["predicted_label"]])

    return output.getvalue()
