"""Synthetic tests for FastAPI endpoints."""
import requests

BASE_URL = "http://localhost:8000"


def test_prediction() -> None:
    """Test normal prediction."""
    resp = requests.post(f"{BASE_URL}/predict", json={"text": "I'm so happy"})
    print("Valid:", resp.status_code, resp.json())


def test_invalid_prediction() -> None:
    """Test invalid prediction request."""
    resp = requests.post(f"{BASE_URL}/predict", json={})
    print("Invalid:", resp.status_code, resp.text)


def test_feedback() -> None:
    """Test feedback submission."""
    payload = {
        "input_hash": "123abc",
        "prediction": "joy",
        "actual_label": "joy",
        "rating": 5,
    }
    resp = requests.post(f"{BASE_URL}/feedback", json=payload)
    print("Feedback:", resp.status_code, resp.json())


if __name__ == "__main__":
    test_prediction()
    test_invalid_prediction()
    test_feedback()
