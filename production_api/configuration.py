import os

MODEL_DIR: str = str(os.environ.get("MODEL_DIR", "./models"))
CURRENT_MODEL_PATH: str = str(os.environ.get("CURRENT_MODEL_PATH", "./models/model.sav"))

MIN_DESC_LEN: int = int(os.environ.get("MIN_DESC_LEN", 40))
MIN_PREDICTING_PROBA: float = float(os.environ.get("MIN_PREDICTING_PROBA", 0.5))

API_HOST: str = str(os.environ.get("API_HOST", "0.0.0.0"))
API_PORT: str = str(os.environ.get("API_PORT", "5000"))
