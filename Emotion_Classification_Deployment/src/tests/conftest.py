"""
Global fixtures & lightweight stub-modules shared by all tests.
This file is auto-discovered by pytest.
"""
from __future__ import annotations
from fastapi.testclient import TestClient
import importlib.machinery
import sys
import io
import tempfile
import builtins
import types
import os
import pytest

# 0) Environment flags – keep Transformers from importing heavy stuff
os.environ.setdefault("TRANSFORMERS_NO_TORCH", "1")   # skip real torch
os.environ.setdefault("TRANSFORMERS_NO_TF",   "1")   # skip TensorFlow
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


# 1) Tiny fake "torch" implementation
if "torch" not in sys.modules:
    torch_fake = types.ModuleType("torch")

    # minimalist tensor-like helpers
    torch_fake.tensor = lambda x, **kw: x
    torch_fake.softmax = lambda x, dim=1: x

    class _FakeScalar(int):
        def item(self): return int(self)
    torch_fake.argmax = lambda x, dim=0: _FakeScalar(0)

    class _NoGrad:
        def __enter__(self): return None
        def __exit__(self, *exc): return False
    torch_fake.no_grad = lambda: _NoGrad()

    # ---- fake cuda submodule ----
    cuda_mod = types.ModuleType("torch.cuda")
    cuda_mod.is_available = lambda: False
    torch_fake.cuda = cuda_mod
    sys.modules["torch.cuda"] = cuda_mod

    torch_fake.__spec__ = importlib.machinery.ModuleSpec(
        name="torch", loader=None
    )
    sys.modules["torch"] = torch_fake


# 2) Tiny fake "transformers" (AutoTokenizer / AutoModel) stubs
if "transformers" not in sys.modules:
    tf_fake = types.ModuleType("transformers")

    # fake tokenizer
    class _FakeTokenizer:
        def __call__(self, text, **kw):
            return {"input_ids": [[0]], "attention_mask": [[1]]}

    # fake model (returns fixed logits)
    class _FakeModel:
        def __call__(self, **kw):
            import torch  # our stub
            return types.SimpleNamespace(
                logits=torch.tensor([[0.1, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0]])
            )

        def eval(self):
            return self

    tf_fake.AutoTokenizer = types.SimpleNamespace(
        from_pretrained=lambda *a, **k: _FakeTokenizer()
    )
    tf_fake.AutoModelForSequenceClassification = types.SimpleNamespace(
        from_pretrained=lambda *a, **k: _FakeModel()
    )

    tf_fake.__spec__ = importlib.machinery.ModuleSpec(
        name="transformers", loader=None
    )
    sys.modules["transformers"] = tf_fake


# 3) Tiny fake "whisper" model (for transcript endpoints)
if "whisper" not in sys.modules:
    whisper_fake = types.ModuleType("whisper")

    class _FakeWhisperModel:
        def transcribe(self, path):          # always returns same text
            return {"text": "This is a fake transcript."}

    whisper_fake.load_model = lambda name="base": _FakeWhisperModel()
    whisper_fake.__spec__ = importlib.machinery.ModuleSpec(
        name="whisper", loader=None
    )
    sys.modules["whisper"] = whisper_fake


# 4) FastAPI TestClient fixture for FastApi_deploying/main.py
"""
Global fixtures for FastAPI tests.
They redirect any write to /app/... into a temp file so CI's read-only
file-system does not break the endpoints.
"""


# ---------- FastAPI client (import AFTER stubs are ready) ----------
@pytest.fixture(scope="session")
def fastapi_client():
    """Reusable TestClient for FastApi_deploying/main.py."""
    from FastApi_deploying import main as fastapi_app
    return TestClient(fastapi_app.app)


# ---------- Patch writes to /app during *all* tests ----------
@pytest.fixture(autouse=True)
def patch_app_dir(monkeypatch, tmp_path):
    """Silence os.makedirs('/app/...') and reroute open('/app/...')."""
    monkeypatch.setattr(os, "makedirs", lambda *a, **k: None)

    real_open = builtins.open

    def _fake_open(path, mode="r", *args, **kwargs):
        path = str(path)
        if path.startswith("/app/"):
            # Write → real temp file, Read → empty buffer
            if "w" in mode or "a" in mode:
                return tempfile.NamedTemporaryFile(delete=False, mode=mode)
            return io.StringIO("")
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", _fake_open)
