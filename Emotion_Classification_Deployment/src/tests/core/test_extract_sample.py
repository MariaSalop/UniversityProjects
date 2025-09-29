import pandas as pd
import extract_sample as ex


def test_load_and_sample_dataset(tmp_path):
    # build dummy dataset with 3 classes
    df = pd.DataFrame(
        {"text": [f"s{i}" for i in range(6)],
         "predicted_emotion": ["a", "b", "c", "a", "b", "c"]}
    )
    src = tmp_path / "src.csv"
    dst = tmp_path / "dst.csv"
    df.to_csv(src, index=False)

    ex.load_and_sample_dataset(src, dst, n=1)
    out = pd.read_csv(dst)
    assert len(out) == 3                              # 1 per class
    assert set(out.predicted_emotion) == {"a", "b", "c"}


def test_create_readme(tmp_path):
    readme = tmp_path / "README.md"
    ex.create_readme(readme)
    text = readme.read_text()
    assert "Sample Dataset" in text
    # sanity-check seven emotions mentioned
    for emo in ["anger", "joy", "optimism", "sadness",
                "surprise", "fear", "disgust"]:
        assert emo in text


def test_main_invokes_helpers(monkeypatch, tmp_path):
    """Run ex.main() in an isolated CWD and ensure helpers are called."""
    # keep track of calls
    called = {"load": False, "readme": False}

    def _fake_load(*a, **kw):
        called["load"] = True

    def _fake_readme(*a, **kw):
        called["readme"] = True

    monkeypatch.chdir(tmp_path)                # everything under tmp
    monkeypatch.setattr(ex, "load_and_sample_dataset", _fake_load)
    monkeypatch.setattr(ex, "create_readme", _fake_readme)

    ex.main()

    assert called["load"] and called["readme"]
    assert (tmp_path / "data").is_dir()
