# tests/test_main.py
import logging


def setup_logger():
    logger = logging.getLogger("test_logger")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        file_handler = logging.FileHandler("log.txt")
        console_handler = logging.StreamHandler()

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger


def test_sample():
    logger = setup_logger()
    logger.info("Running sample test...")
    assert 1 + 1 == 2
    logger.info("Sample test completed successfully.")
