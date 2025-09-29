import pytest
from project_nlp8.model import load_model, predict


def test_loader_raises_if_missing():
    missing = "non_existing_checkpoint.pt"
    with pytest.raises(FileNotFoundError):
        load_model(missing)


def test_predict_is_deterministic(tmp_path):
    # 1) Create a temporary checkpoint file
    ckpt = tmp_path / "dummy.pt"
    ckpt.write_text("weights")      # file content doesn’t matter for DummyModel
    model = load_model(str(ckpt))
    # 2) Call predict twice with the same input string
    label1, conf1 = predict(model, "I am sad")
    label2, conf2 = predict(model, "I am sad")
    # 3) Expect identical (label, confidence) tuples on both calls
    assert (label1, conf1) == (label2, conf2)
