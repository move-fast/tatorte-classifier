import os
from dotenv import load_dotenv

load_dotenv()

MODEL_DIR: str = str(os.getenv("MODEL_DIR"))
CURRENT_MODEL_PATH: str = str(os.getenv("CURRENT_MODEL_PATH"))

MIN_DESC_LEN = int(os.getenv("MIN_DESC_LEN"))
MIN_PREDICTING_PROBA = float(os.getenv("MIN_PREDICTING_PROBA"))

API_HOST: str = str(os.getenv("API_HOST"))
API_PORT: int = int(os.getenv("API_PORT"))

MONGO_AUTH: str = str(os.getenv("MONGO_AUTH"))
MONGO_URL: str = str(os.getenv("MONGO_URL"))
MONGO_PORT: str = str(os.getenv("MONGO_PORT"))

TEMPLATE_FOLDER: str = str(os.getenv("TEMPLATE_FOLDER"))
