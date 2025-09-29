from click.testing import CliRunner
import pandas as pd
import cli_no_emotion as cli


# ---------- tiny stub ---------- #
def _fake_predict(txt):
    return txt, 1, "joy", [{"label": "joy", "score": 1.0}]


# ---------- choose_device ---------- #
def test_choose_device_cpu(monkeypatch):
    monkeypatch.setattr(cli.typer, "prompt", lambda m: "cpu")
    assert cli.choose_device() == "cpu"


def test_choose_device_gpu_fallback(monkeypatch):
    calls = iter(["bad", "gpu"])
    monkeypatch.setattr(cli.typer, "prompt", lambda m: next(calls))
    monkeypatch.setattr(cli.torch.cuda, "is_available", lambda: False)
    assert cli.choose_device() == "cpu"


def test_choose_device_gpu_ok(monkeypatch):
    monkeypatch.setattr(cli.typer, "prompt", lambda m: "gpu")
    monkeypatch.setattr(cli.torch.cuda, "is_available", lambda: True)
    assert cli.choose_device() == "cuda"


# ---------- init_model ---------- #
def test_init_model(monkeypatch):
    class _M:
        def to(self, dev):
            self.dev = dev
            return self
    monkeypatch.setattr(cli.AutoModelForSequenceClassification,
                        "from_pretrained", lambda p: _M())
    monkeypatch.setattr(cli.AutoTokenizer,
                        "from_pretrained", lambda p: "tok")
    m, t = cli.init_model("cpu")
    assert getattr(m, "dev", None) == "cpu" and t == "tok"


# ---------- CLI commands ---------- #
def test_cli_predict(monkeypatch):
    monkeypatch.setattr(cli, "predict_emotion", _fake_predict)
    res = CliRunner().invoke(
        cli.app,
        ["predict", "hi"],
        prog_name="cli-no-emotion",
    )
    assert res.exit_code == 0 and "joy" in res.output.lower()


def test_cli_batch_success(tmp_path, monkeypatch):
    csv = tmp_path / "input.csv"
    pd.DataFrame({"text": ["x"]}).to_csv(csv, index=False)
    monkeypatch.setattr(cli, "predict_emotion", _fake_predict)
    monkeypatch.setattr(cli, "tqdm", lambda it, **kw: it)   # silence bar
    res = CliRunner().invoke(
        cli.app,
        ["batch-predict", str(csv), "--text-column", "text"],
        prog_name="cli-no-emotion",
    )
    assert res.exit_code == 0
    assert (tmp_path / "input_v2.csv").exists()


def test_cli_batch_file_not_found():
    res = CliRunner().invoke(
        cli.app, ["batch-predict", "nope.csv"], prog_name="cli-no-emotion"
    )
    assert res.exit_code == 0
    assert "file not found" in res.output.lower()


def test_cli_batch_missing_column(tmp_path, monkeypatch):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"other": ["x"]}).to_csv(bad, index=False)

    monkeypatch.setattr(cli, "predict_emotion", _fake_predict)
    res = CliRunner().invoke(
        cli.app,
        ["batch-predict", str(bad), "--text-column", "text"],
        prog_name="cli-no-emotion",
    )
    assert res.exit_code == 0
    assert "column 'text' not found" in res.output.lower()


def test_cli_model_info():
    assert CliRunner().invoke(
        cli.app,
        ["model-info"],
        prog_name="cli-no-emotion",
    ).exit_code == 0
