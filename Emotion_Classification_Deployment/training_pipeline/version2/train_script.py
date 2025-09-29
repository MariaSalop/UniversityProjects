import argparse
import os
import torch
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
    AutoConfig
)
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score
from azureml.core import Run
from azureml.core.model import Model
import pandas as pd
 
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    return parser.parse_args()
 
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted")
    }
 
def load_local_csvs(train_path, val_path):
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    # Rename the label column
    train_df = train_df.rename(columns={"predicted_emotion": "label"})
    val_df = val_df.rename(columns={"predicted_emotion": "label"})
    # Drop unnecessary columns (optional)
    train_df = train_df[["text", "label"]]
    val_df = val_df[["text", "label"]]
    # Convert label strings to ids
    label2id = {label: i for i, label in enumerate(sorted(train_df["label"].unique()))}
    train_df["label"] = train_df["label"].map(label2id)
    val_df["label"] = val_df["label"].map(label2id)
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    return DatasetDict({"train": train_dataset, "validation": val_dataset})
 
def resolve_model_dir(model_dir):
    # Check if model files are in a 'model' subdir
    contents = os.listdir(model_dir)
    required_files = [
        "config.json", "merges.txt", "model.safetensors",
        "special_tokens_map.json", "tokenizer_config.json", "tokenizer.json", "vocab.json"
    ]
    model_subdir = os.path.join(model_dir, "model")
    if "model" in contents and all(f in os.listdir(model_subdir) for f in required_files):
        print("DEBUG: Using model subdir:", model_subdir)
        return model_subdir
    if all(f in contents for f in required_files):
        print("DEBUG: Model files found directly in", model_dir)
        return model_dir
    raise RuntimeError(f"Could not find expected model files in {model_dir} or its subfolders.")
 
def main():
    args = parse_args()
    run = Run.get_context()
 
    # Resolve the registered model path inside AzureML
    if run.id != "OfflineRun":
        print("DEBUG: Resolving registered model path in AzureML run context")
        args.model_dir = Model.get_model_path(args.model_dir)
        print("DEBUG: Resolved model_dir =", args.model_dir)
 
    print("DEBUG: args.model_dir =", args.model_dir)
    print("DEBUG: Does args.model_dir exist?", os.path.exists(args.model_dir))
    if os.path.exists(args.model_dir):
        print("DEBUG: Contents of args.model_dir:", os.listdir(args.model_dir))
    else:
        print("DEBUG: args.model_dir DOES NOT EXIST")
 
    # Find the actual model dir to load from
    model_dir_for_loading = resolve_model_dir(args.model_dir)
 
    tokenizer = RobertaTokenizer.from_pretrained(model_dir_for_loading)
    config = AutoConfig.from_pretrained(model_dir_for_loading)
    model = RobertaForSequenceClassification.from_pretrained(
        model_dir_for_loading,
        config=config
)
 
    datasets = load_local_csvs(args.train_data, args.val_data)
 
    def tokenize(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True)
 
    tokenized_datasets = datasets.map(tokenize, batched=True)
    tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
 
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=1,
        save_strategy="no",
        load_best_model_at_end=False,
        metric_for_best_model="accuracy",
)
 
 
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        compute_metrics=compute_metrics,
    )
 
    trainer.train()
    eval_metrics = trainer.evaluate()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
 
    for k, v in eval_metrics.items():
        run.log(k, v)
 
if __name__ == "__main__":
    main()
