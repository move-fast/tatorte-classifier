import os
from dotenv import load_dotenv

load_dotenv()

MODEL_DIR: str = str(os.getenv("MODEL_DIR"))
CURRENT_MODEL_PATH: str = str(os.getenv("CURRENT_MODEL_PATH"))

MIN_DESC_LEN: int = int(os.getenv("MIN_DESC_LEN"))
MIN_PREDICTING_PROBA: float = float(os.getenv("MIN_PREDICTING_PROBA"))

API_HOST: str = str(os.getenv("API_HOST"))
API_PORT: str = str(os.getenv("API_PORT"))

DATA_X_PATH: str = str(os.getenv("DATA_X_PATH"))
DATA_Y_PATH: str = str(os.getenv("DATA_Y_PATH"))
