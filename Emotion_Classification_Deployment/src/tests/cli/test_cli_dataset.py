from click.testing import CliRunner
import pandas as pd
import pytest
import cli_dataset


# ---------- helpers ---------- #
@pytest.fixture
def dummy_nlp(monkeypatch):
    class _Tok:  # fake HF tokenizer
        def __call__(self, txt, **kw):
            return {"input_ids": [[0]], "attention_mask": [[1]]}

    class _Model:  # fake HF model
        def __call__(self, **kw):
            import torch
            logits = torch.tensor([[1, 2, 0.5, 3, -1, 0, 0.2]])
            return type("Out", (), {"logits": logits})

    monkeypatch.setattr(cli_dataset, "_tokenizer", _Tok())
    monkeypatch.setattr(cli_dataset, "_model", _Model())


# ---------- unit ---------- #
def test_predict_emotion_returns_structure(dummy_nlp):
    _, idx, label, scores = cli_dataset.predict_emotion("hello")
    assert label == cli_dataset.LABELS[idx]
    assert len(scores) == len(cli_dataset.LABELS)


# ---------- CLI: success paths ---------- #
def test_cli_test_cmd(dummy_nlp):
    out = CliRunner().invoke(
        cli_dataset.app, ["test", "Hello"], prog_name="cli-dataset"
    ).output
    assert "hello" in out.lower()


def test_cli_predict_cmd(monkeypatch):
    monkeypatch.setattr(
        cli_dataset, "predict_emotion",
        lambda txt: (txt, 3, "surprise", [{"label": "surprise", "score": .9}]),
    )
    res = CliRunner().invoke(
        cli_dataset.app, ["predict", "Wow!"],
        prog_name="cli-dataset",
    )
    assert res.exit_code == 0 and "surprise" in res.output.lower()


def test_cli_batch_predict_success(tmp_path, monkeypatch):
    csv = tmp_path / "data.csv"
    pd.DataFrame({"text": ["a", "b"]}).to_csv(csv, index=False)
    monkeypatch.setattr(cli_dataset, "predict_emotion",
                        lambda t: (t, 0, "anger", []))
    res = CliRunner().invoke(
        cli_dataset.app, ["batch-predict", str(csv)],
        prog_name="cli-dataset",
    )
    assert res.exit_code == 0
    assert (tmp_path / "data_predicted.csv").exists()


# ---------- CLI: error paths ---------- #
def test_cli_batch_predict_file_not_found():
    res = CliRunner().invoke(
        cli_dataset.app, ["batch-predict", "missing.csv"],
        prog_name="cli-dataset",
    )
    assert res.exit_code == 0
    assert "file not found" in res.output.lower()


def test_cli_batch_predict_missing_column(tmp_path, dummy_nlp):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"other": ["x"]}).to_csv(bad, index=False)
    res = CliRunner().invoke(
        cli_dataset.app, ["batch-predict", str(bad), "--text-column", "text"],
        prog_name="cli-dataset",
    )
    assert res.exit_code == 0
    assert "column 'text' not found" in res.output.lower()
