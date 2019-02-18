from flask import Flask, jsonify, request
import numpy as np

from get_prediction import get_predictions, load_model
from preprocess_data import DataPreprocessor
from configuration import API_HOST, API_PORT

app = Flask("tatorte_api")
model = load_model()
preprocessor = DataPreprocessor()


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

    key = [
        {"key": 0, "name": "Feuer"},
        {"key": 1, "name": "Überfall/Körperverletzung"},
        {"key": 2, "name": "Mord"},
        {"key": 3, "name": "Drogen"},
        {"key": 4, "name": "Unfall"},
    ]
    return jsonify(key)


if __name__ == "__main__":
    app.run(host=API_HOST, port=API_PORT)
