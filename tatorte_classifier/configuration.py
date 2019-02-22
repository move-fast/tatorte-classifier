import os
from dotenv import load_dotenv

load_dotenv()

MODEL_DIR: str = str(os.getenv("MODEL_DIR"))
CURRENT_MODEL_PATH: str = str(os.getenv("CURRENT_MODEL_PATH"))

MIN_DESC_LEN = int(os.getenv("MIN_DESC_LEN"))
MIN_PREDICTING_PROBA = float(os.getenv("MIN_PREDICTING_PROBA"))

API_HOST: str = str(os.getenv("API_HOST"))
API_PORT: int = int(os.getenv("API_PORT"))
API_DEBUG: bool = bool(os.getenv("API_DEBUG"))

MONGODB_URI: str = str(os.getenv("MONGODB_URI"))
TEMPLATE_FOLDER: str = str(os.getenv("TEMPLATE_FOLDER"))
