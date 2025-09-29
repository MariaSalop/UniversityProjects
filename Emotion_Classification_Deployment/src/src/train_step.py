# src/train_step.py
"""CLI entry-point for Azure ML retraining pipeline.

The script
1. loads train / validation CSV files that already contain the *clean_text* column;
2. tokenises texts with a pre-trained RoBERTa tokenizer;
3. fine-tunes `RobertaForSequenceClassification`;
4. writes the trained model to `--model_out` (folder) **and** dumps
   evaluation metrics to `--metrics_out` (JSON).

Example (local):
python train_step.py \
       --train_csv data/clean/train.csv \
       --val_csv   data/clean/val.csv \
       --pretrained_dir models/roberta_base \
       --model_out outputs/model \
       --metrics_out outputs/metrics.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
    AutoConfig,
)
# Optional but handy inside AML runs
try:
    from azureml.core import Run  # type: ignore
except ModuleNotFoundError:  # local/offline
    class _OfflineRun:  # minimal stub
        id = "OfflineRun"
        @staticmethod
        def log(*_a, **_kw): ...
    Run = _OfflineRun  # type: ignore


def _compute_metrics(eval_pred):
    """HF Trainer callback -> dict of scalar metrics."""
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1":       f1_score(labels, preds, average="weighted"),
    }


def _load_csvs(train_path: Path, val_path: Path) -> DatasetDict:
    """Read CSVs, map text→ids label→int → HF DatasetDict."""
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    # Expect 'clean_text' and 'label' columns (or 'predicted_emotion')
    if "label" not in train_df.columns:
        train_df = train_df.rename(columns={"predicted_emotion": "label"})
        val_df = val_df.rename(columns={"predicted_emotion": "label"})

    # Convert string labels to ids
    lbl2id = {l: i for i, l in enumerate(sorted(train_df["label"].unique()))}
    train_df["label"] = train_df["label"].map(lbl2id)
    val_df["label"] = val_df["label"].map(lbl2id)

    train_ds = Dataset.from_pandas(train_df[["clean_text", "label"]])
    val_ds = Dataset.from_pandas(val_df[["clean_text", "label"]])
    return DatasetDict({"train": train_ds, "validation": val_ds})


def main() -> None:
    p = argparse.ArgumentParser(description="Fine-tune RoBERTa on emotion dataset")
    p.add_argument("--train_csv",    type=Path, required=True)
    p.add_argument("--val_csv",      type=Path, required=True)
    p.add_argument("--pretrained_dir", type=Path, required=True,
                   help="Local dir or registered AML model containing RoBERTa files")
    p.add_argument("--model_out",    type=Path, required=True)
    p.add_argument("--metrics_out",  type=Path, required=True)
    p.add_argument("--epochs",       type=int, default=1)
    args = p.parse_args()

    run = Run.get_context()

    # If running inside AML, resolve registered model
    if getattr(run, "id", "OfflineRun") != "OfflineRun":
        # In AML runs, pretrained_dir is already the local mount path for the model asset,
        # so we do not need to call Model.get_model_path here.
        # args.pretrained_dir remains the mounted directory.
        pass

    # ↳ May be nested in a "model" subfolder (helper from existing script)
    required = {"config.json", "merges.txt", "pytorch_model.bin", "vocab.json"}
    if not any((args.pretrained_dir / p).exists() for p in required):
        sub = args.pretrained_dir / "model"
        if sub.exists():
            args.pretrained_dir = sub

    tokenizer = RobertaTokenizer.from_pretrained(str(args.pretrained_dir))
    config = AutoConfig.from_pretrained(str(args.pretrained_dir))
    model = RobertaForSequenceClassification.from_pretrained(
        str(args.pretrained_dir),
        config=config,
        ignore_mismatched_sizes=True,
    )

    datasets = _load_csvs(args.train_csv, args.val_csv)

    def _tok(batch):
        return tokenizer(batch["clean_text"], padding="max_length", truncation=True)

    tokenised = datasets.map(_tok, batched=True)
    tokenised.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "label"],
    )

    train_args = TrainingArguments(
        output_dir=str(args.model_out),
        evaluation_strategy="epoch",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        save_strategy="no",
        load_best_model_at_end=False,
        metric_for_best_model="accuracy",
        logging_steps=50,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=tokenised["train"],
        eval_dataset=tokenised["validation"],
        compute_metrics=_compute_metrics,
    )

    trainer.train()
    eval_metrics = trainer.evaluate()

    # Save artifacts
    args.model_out.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.model_out))
    tokenizer.save_pretrained(str(args.model_out))

    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.metrics_out, "w") as fp:
        json.dump(eval_metrics, fp, indent=2)

    # Log to AML run (if any)
    for k, v in eval_metrics.items():
        run.log(k, float(v))


if __name__ == "__main__":
    main()
