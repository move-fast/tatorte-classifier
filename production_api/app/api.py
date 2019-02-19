import datetime
import os

import numpy as np
import pymongo
from bson.json_util import dumps
from bson.objectid import ObjectId
from flask import Flask, jsonify, request

from configuration import (
    API_HOST,
    API_PORT,
    MONGO_PASSWORD,
    MONGO_PORT,
    MONGO_URL,
    MONGO_USER,
    CURRENT_MODEL_PATH,
    MODEL_DIR,
)
from get_prediction import get_predictions, load_model
from preprocess_data import DataPreprocessor
from train_model import main as train_model

# TODO: Add Error checking
# TODO: Change function names
# TODO: Ask if conf_matrix should also be returned

app = Flask("tatorte_api")
model = load_model()
preprocessor = DataPreprocessor()
client = pymongo.MongoClient(
    "mongodb://{}:{}@{}:{}/tatorte-db".format(MONGO_USER, MONGO_PASSWORD, MONGO_URL, MONGO_PORT)
)
db = client["tatorte-db"]
texts = db["texts"]


@app.route("/get_prediction", methods=["POST"])
def get_preds():
    """Get predictions and probabilitys
    Input:
        {
            "data": "This is a test"
            "parameters": {
                "max_categories": 3
            }
        }
    Returns:
        {"predictions":
            [{
                "category": 2,
                "probability": 0.57
            }, {
                "category": 0,
                "probability": 0.36
            }, {
                "category": 3,
                "probability": 0.12
            }]
        }
    """
    request_json = request.get_json()
    max_categories = request_json["parameters"]["max_categories"]
    desc = request_json["data"]
    desc = preprocessor(desc)
    preds = np.asarray(get_predictions([desc], model, max_categories)).T
    preds = [{"category": pred_idx, "probability": pred} for pred_idx, pred in preds]
    return jsonify({"predictions": preds})


@app.route("/categories", methods=["GET"])
def get_key():
    """Returns the key for translating class_numbers to text

    Returns:
        json -- A json object where key is the class_number and the value is the
                corresponding class_text
    """

    keys = [
        {"key": 0, "name": "Feuer"},
        {"key": 1, "name": "Mord"},
        {"key": 2, "name": "Überfall/Körperverletzung"},
        {"key": 3, "name": "Unfall"},
        {"key": 4, "name": "Drogen"},
    ]
    return jsonify(keys)


@app.route("/texts", methods=["GET"])
def get_texts():
    return dumps(texts.find())


@app.route("/text/<id>", methods=["GET"])
def get_text(id):
    return dumps(texts.find_one({"_id": ObjectId(id)}))


@app.route("/create_text", methods=["POST"])
def create_text():
    """
    Input:
        {
            "data": "This is a test",
            "categories": [3, 2]
        }
    
    Returns:
        <id>
    """

    request_json = request.get_json()
    text_created_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    text = {
        "data": request_json["data"],
        "time_created": text_created_date,
        "time_modified": text_created_date,
        "categories": request_json["categories"],
    }
    text_id = texts.insert_one(text).inserted_id
    return str(text_id)


@app.route("/update_categories", methods=["POST"])
def change_text():
    """
    Input:
        {
            "categories": [3, 2],
            "id": hd897e9289a
        }
        
    Returns:
        <id>
    """
    try:
        request_json = request.get_json()
        id = request_json["id"]
        texts.update_one(
            {"_id": ObjectId(id)},
            {
                "$set": {
                    "categories": request_json["categories"],
                    "time_modified": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
            },
            upsert=False,
        )
        return jsonify({"success": True})
    except:
        return jsonify({"success": False})


@app.route("/train_model", methods=["POST"])
def new_model():
    """
    Input:
        {
            "clf": "sgd",
            "clf_params": {
                "alpha": 1e-6,
                "max_iter": 100, 
                "loss": "log", 
                "penalty": "l2"
            },
            "vect_params": {
                "ngram_range": [1, 4]
            },
            "test_size": 0.2,
            "values_per_category": 900
        }
    Returns:

    """

    request_json = request.get_json()
    id, train_acc, test_acc, conf_matrix = train_model(**request_json)
    return jsonify({"id": id, "train_acc": train_acc, "test_acc": test_acc})


@app.route("/change_model/<id>", methods=["GET"])
def change_model(id):
    try:
        os.system("rm -f " + CURRENT_MODEL_PATH)
        os.system("cp {}/model-{}.sav {}".format(MODEL_DIR, id, CURRENT_MODEL_PATH))
        model = load_model()
        return jsonify({"success": True})
    except:
        return jsonify({"success": False})


if __name__ == "__main__":
    app.run(debug=True, host=API_HOST, port=API_PORT)
