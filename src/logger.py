import logging
import os
from datetime import datetime

# Define log file name and path
LOG_FILE = f"{datetime.now().strftime('%Y_%m_%d__%H_%M_%S')}.log"
logs_path = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_path, exist_ok=True)  # Create the logs directory if it doesn't exist

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# Set up the global logging configuration
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# Test logger functionality
if __name__ == "__main__":
    try:
        logging.info("Logging has started successfully.")
        # Simulating an exception for testing purposes
        raise ValueError("This is a test exception for the logger.")
    except Exception as e:
        logging.error("An exception occurred", exc_info=True)
