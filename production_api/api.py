from flask import Flask, jsonify, request
import numpy as np

# stemmed_words must be imported for loading the Model
from get_prediction import stemmed_words, get_predictions, load_model


app = Flask("tatorte_api")
model = load_model()


@app.route("/get_prediction/<n_classes>", methods=["POST"])
def get_preds(n_classes):
    """Get predictions and probabilitys
    Input:
        {
            "data": "This is a test"
        }
    Returns:
        {
            "classes": [2, 0],
            "probs": [0.57, 0.36]
        }
    """

    desc = request.get_json()["data"]
    pred_idx, pred = get_predictions([desc], model, int(n_classes))
    preds = {"class": pred_idx, "probs": pred}
    return jsonify(preds)


@app.route("/get_key", methods=["GET"])
def get_key():
    """Returns the key for translating class_numbers to text

    Returns:
        json -- A json object where key is the class_number and the value is the
                corresponding class_text
    """

    key = {0: "Feuer", 1: "Überfall/Körperverletzung", 2: "Mord", 3: "Drogen", 4: "Unfall"}
    return jsonify(key)


if __name__ == "__main__":
    app.run(debug=True)
