import logging
from locust import HttpUser, task, between

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("locust_test")

class APIUser(HttpUser):
    wait_time = between(1, 2)  # seconds between tasks

    @task
    def predict_emotion(self):
        payload = {"text": "I am happy with this test!"}
        logger.info("Sending prediction request with payload: %s", payload)
        response = self.client.post("/predict", json=payload)
        logger.info("Received response: %s", response.text)
        if response.status_code != 200:
            logger.error("Error! Status code: %s | Response: %s", response.status_code, response.text)
