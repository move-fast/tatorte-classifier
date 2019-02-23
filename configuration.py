import os
from dotenv import load_dotenv

load_dotenv()

MODEL_DIR: str = str(os.getenv("MODEL_DIR"))
CURRENT_MODEL_PATH: str = str(os.getenv("CURRENT_MODEL_PATH"))

MIN_DESC_LEN = int(os.getenv("MIN_DESC_LEN"))
MIN_PREDICTING_PROBA = float(os.getenv("MIN_PREDICTING_PROBA"))

HOST: str = str(os.getenv("HOST"))
PORT: str = str(os.getenv("PORT"))

MONGODB_URI: str = str(os.getenv("MONGODB_URI"))
TEMPLATE_FOLDER: str = str(os.getenv("TEMPLATE_FOLDER"))
FLASK_MODE: str = str(os.getenv("FLASK_MODE"))

# Flask conf
class Config:
    DEBUG = False
    TESTING = False


class DevelopementConfig(Config):
    DEBUG = True


ProductionConfig = Config  # At the moment nothing differs to normal config
