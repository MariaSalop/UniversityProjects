from FastApi_deploying.services import model_service
import torch


def test_predict_returns_label_and_probs(monkeypatch):
    # guaranteed predictable output
    monkeypatch.setattr(model_service, "_load_pipeline", lambda: (None, None))

    def _one_hot(x, dim=1):
        return torch.tensor([[1.0] + [0.0]*(len(model_service.LABELS)-1)])

    monkeypatch.setattr(model_service.torch, "softmax", _one_hot)

    label, probs = model_service.predict("hello")
    assert label == model_service.LABELS[0]
    assert len(probs) == len(model_service.LABELS)
