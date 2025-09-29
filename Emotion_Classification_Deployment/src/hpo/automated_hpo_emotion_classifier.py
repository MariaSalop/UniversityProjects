import os
import pandas as pd
import numpy as np
import torch
import optuna
import joblib
from datasets import Dataset, Value
from transformers import (
    TrainingArguments, Trainer, RobertaTokenizerFast,
    RobertaForSequenceClassification, set_seed
)
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import argparse

# Set random seeds for reproducibility
def set_all_seeds(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed)

def load_and_balance_data(csv_path, max_per_emotion=200, seed=42):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["text", "emotion"])
    df = (
        df.groupby("emotion", group_keys=False)
        .apply(lambda g: g.sample(n=min(len(g), max_per_emotion), random_state=seed))
        .reset_index(drop=True)
    )
    return df

def encode_labels(df):
    unique_emotions = sorted(df["emotion"].unique())
    label2id = {emotion: idx for idx, emotion in enumerate(unique_emotions)}
    id2label = {idx: emotion for emotion, idx in label2id.items()}
    df["label"] = df["emotion"].map(label2id)
    return df, label2id, id2label

def tokenize_datasets(train_df, test_df, tokenizer_name="roberta-base"):
    tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_name)
    def tokenize(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)
    train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
    test_dataset = Dataset.from_pandas(test_df.reset_index(drop=True))
    train_dataset = train_dataset.map(tokenize, batched=True, num_proc=4)
    test_dataset = test_dataset.map(tokenize, batched=True, num_proc=4)
    train_dataset = train_dataset.remove_columns(["text", "emotion"])
    test_dataset = test_dataset.remove_columns(["text", "emotion"])
    train_dataset = train_dataset.cast_column("label", Value("int64"))
    test_dataset = test_dataset.cast_column("label", Value("int64"))
    return train_dataset, test_dataset, tokenizer

def compute_metrics(pred):
    preds = np.argmax(pred.predictions, axis=1)
    labels = pred.label_ids
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted")
    }

def model_init(num_labels, model_name="roberta-base"):
    return RobertaForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

def objective(trial, train_dataset, test_dataset, tokenizer, num_labels):
    args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",   # Ensure this matches save_strategy!
        save_strategy="epoch",
        learning_rate=trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
        per_device_train_batch_size=trial.suggest_categorical("batch_size", [8, 16, 32]),
        num_train_epochs=trial.suggest_int("epochs", 2, 4),
        weight_decay=trial.suggest_float("weight_decay", 0.0, 0.3),
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_dir="./logs",
        disable_tqdm=True,
        seed=42
    )

    trainer = Trainer(
        model_init=lambda: model_init(num_labels),
        args=args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    metrics = trainer.evaluate()
    trial.report(metrics["eval_accuracy"], step=0)
    if trial.should_prune():
        raise optuna.TrialPruned()
    return metrics["eval_accuracy"]

def main(
    csv_path,
    n_trials=20,
    max_per_emotion=200,
    random_seed=42,
    study_path="optuna_emotion_hpo_study.pkl"
):
    set_all_seeds(random_seed)

    # 1. Data loading and balancing
    df = load_and_balance_data(csv_path, max_per_emotion=max_per_emotion, seed=random_seed)
    df, label2id, id2label = encode_labels(df)

    # 2. Split
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["label"],
        random_state=random_seed
    )

    # 3. Tokenization
    train_dataset, test_dataset, tokenizer = tokenize_datasets(train_df, test_df)

    # 4. HPO with Optuna (resumable)
    def optuna_objective(trial):
        return objective(
            trial,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            tokenizer=tokenizer,
            num_labels=len(label2id)
        )

    if os.path.exists(study_path):
        study = joblib.load(study_path)
        print(f"Resuming existing Optuna study with {len(study.trials)} trials.")
    else:
        study = optuna.create_study(direction="maximize")
        print("Created new Optuna study.")

    study.optimize(optuna_objective, n_trials=n_trials)
    joblib.dump(study, study_path)
    best_params = study.best_params
    print("Best Hyperparameters:", best_params)

    # 5. Final training
    final_args = TrainingArguments(
        output_dir="./final_model",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=best_params["learning_rate"],
        per_device_train_batch_size=best_params["batch_size"],
        num_train_epochs=best_params["epochs"],
        weight_decay=best_params["weight_decay"],
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_dir="./logs",
        logging_steps=100,
        seed=random_seed
    )

    final_trainer = Trainer(
        model_init=lambda: model_init(len(label2id)),
        args=final_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    final_trainer.train()
    final_trainer.save_model("./final_model/checkpoint-last")
    metrics = final_trainer.evaluate()
    print("Final evaluation:", metrics)
    print("Script completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the input CSV file")
    parser.add_argument("--n_trials", type=int, default=20, help="Number of Optuna trials for HPO")
    parser.add_argument("--max_per_emotion", type=int, default=200, help="Max samples per emotion class")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--study_path", type=str, default="optuna_emotion_hpo_study.pkl", help="Path to save/load Optuna study")
    args = parser.parse_args()

    main(
        csv_path=args.csv_path,
        n_trials=args.n_trials,
        max_per_emotion=args.max_per_emotion,
        random_seed=args.seed,
        study_path=args.study_path
    )
