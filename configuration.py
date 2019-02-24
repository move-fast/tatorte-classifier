import os
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

load_dotenv()

if "MODEL_DIR" not in os.environ:
    logger.warning("MODEL_DIR is not set")
MODEL_DIR: str = str(os.getenv("MODEL_DIR"))

if "CURRENT_MODEL_PATH" not in os.environ:
    logger.warning("CURRENT_MODEL_PATH is not set")
CURRENT_MODEL_PATH: str = str(os.getenv("CURRENT_MODEL_PATH"))

if "MIN_DESC_LEN" not in os.environ:
    logger.warning("MIN_DESC_LEN is not set")
MIN_DESC_LEN = int(os.getenv("MIN_DESC_LEN"))

if "MIN_PREDICTING_PROBA" not in os.environ:
    logger.warning("MIN_PREDICTING_PROBA is not set")
MIN_PREDICTING_PROBA = float(os.getenv("MIN_PREDICTING_PROBA"))

if "HOST" not in os.environ:
    logger.warning("HOST is not set")
HOST: str = str(os.getenv("HOST"))

if "PORT" not in os.environ:
    logger.warning("PORT is not set")
PORT: str = str(os.getenv("PORT"))

if "MONGODB_URI" not in os.environ:
    logger.warning("MONGODB_URI is not set")
MONGODB_URI: str = str(os.getenv("MONGODB_URI"))

if "TEMPLATE_FOLDER" not in os.environ:
    logger.warning("TEMPLATE_FOLDER is not set")
TEMPLATE_FOLDER: str = str(os.getenv("TEMPLATE_FOLDER"))

if "FLASK_MODE" not in os.environ:
    logger.warning("FLASK_MODE is not set")
FLASK_MODE: str = str(os.getenv("FLASK_MODE"))

if "LOG_FILE" not in os.environ:
    logger.warning("LOG_FILE is not set")
LOG_FILE: str = str(os.getenv("LOG_FILE"))

# Flask conf
class Config:
    DEBUG = False
    TESTING = False
    FLASK_LOGGING_EXTRAS = {
        "BLUEPRINT": {
            "FORMAT_NAME": "bp",
            "APP_BLUEPRINT": "<app>",
            "NO_REQUEST_BLUEPRINT": "<not a request>",
        },
        "RESOLVERS": {"categoy": "<unset>", "client": "log_helper.get_client"},
    }


class DevelopementConfig(Config):
    DEBUG = True


ProductionConfig = Config  # At the moment nothing differs to normal config
