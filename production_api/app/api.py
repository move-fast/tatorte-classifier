import datetime
import os

import numpy as np
import pymongo
import requests
from bson.json_util import dumps
from bson.objectid import ObjectId
from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.exceptions import BadRequest

from configuration import (
    API_HOST,
    API_PORT,
    CURRENT_MODEL_PATH,
    MODEL_DIR,
    MONGO_PASSWORD,
    MONGO_PORT,
    MONGO_URL,
    MONGO_USER,
    TEMPLATE_FOLDER,
)
from get_prediction import get_predictions, load_model
from preprocess_data import DataPreprocessor
from train_model import main as train_model

# TODO: Change function names
# TODO: Ask if conf_matrix should also be returned
# TODO: Change filename of returned object by /model/<model_id>
# TODO: Make stuff more consistent
# TODO: Add search function to texts
# TODO: Add Model website

app = Flask("tatorte_api", template_folder=TEMPLATE_FOLDER)
model = load_model()
preprocessor = DataPreprocessor()
client = pymongo.MongoClient(
    "mongodb://{}:{}@{}:{}/tatorte-db".format(MONGO_USER, MONGO_PASSWORD, MONGO_URL, MONGO_PORT)
)
db = client["tatorte-db"]
texts = db["texts"]

#######
# API #
#######
@app.route("/api/get_prediction", methods=["POST"])
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
    try:
        request_json = request.get_json()
        max_categories = request_json["parameters"]["max_categories"]
        desc = request_json["data"]
        desc = preprocessor(desc)
        preds = np.asarray(get_predictions([desc], model, max_categories)).T
        preds = [{"category": pred_idx, "probability": pred} for pred_idx, pred in preds]
        return jsonify({"predictions": preds})
    except Exception as err:
        return BadRequest(str(err))


@app.route("/api/categories", methods=["GET"])
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


@app.route("/api/texts/<start>&<end>", methods=["GET"])
def get_texts(start, end):
    """
    
    Returns:
        [
            {
                "_id": {
                    "$oid": "5c6c1b2573cda500b254404c"
                }, 
                "data": "This is a test. Number 2",
                "time_created": "2019-02-19 15:05:09",
                "time_modified": "2019-02-19 15:18:53",
                "categories": [4, 2]
            }, ...
        ]
    """

    return dumps(texts.find().sort("time_modified", pymongo.DESCENDING)[int(start) : int(end)])


@app.route("/api/text/<text_id>", methods=["GET"])
def get_text(text_id):
    """
    Returns:
        {
            "_id": {
                "$oid": "5c6c1b2573cda500b254404c"
            }, 
            "data": "This is a test. Number 2",
            "time_created": "2019-02-19 15:05:09",
            "time_modified": "2019-02-19 15:18:53",
            "categories": [4, 2]
        }
    """
    try:
        return dumps(texts.find_one({"_id": ObjectId(text_id)}))
    except Exception as err:
        return BadRequest(str(err))


@app.route("/api/create_text", methods=["POST"])
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
    try:
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
    except Exception as err:
        return BadRequest(str(err))


@app.route("/api/change_categories", methods=["POST"])
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
    except Exception as err:
        return BadRequest(str(err))


@app.route("/api/delete_text/<text_id>")
def delete_text(text_id):
    try:
        texts.delete_one({"_id": ObjectId(text_id)})
        return jsonify({"success": True})
    except Exception as err:
        return BadRequest(str(err))


@app.route("/api/train_model", methods=["POST"])
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
    try:
        request_json = request.get_json()
        data = list(texts.find({}, {"data": 1, "categories": 1}))
        data = np.asarray([[i["data"], i["categories"][0]] for i in data]).T
        x, y = data[0], data[1].astype(int)
        id, train_acc, test_acc, _ = train_model(x, y, **request_json)
        return jsonify({"id": id, "train_acc": train_acc, "test_acc": test_acc})
    except Exception as err:
        return BadRequest(str(err))


@app.route("/api/change_model/<model_id>", methods=["GET"])
def change_model(model_id):
    global model
    try:
        os.system("rm -f " + CURRENT_MODEL_PATH)
        os.system("cp {}/model-{}.sav {}".format(MODEL_DIR, model_id, CURRENT_MODEL_PATH))
        model = load_model()
        return jsonify({"success": True})
    except Exception as err:
        return BadRequest(str(err))


@app.route("/api/get_n_texts", methods=["GET"])
def get_n_texts():
    return texts.count()


@app.route("/api/model/<model_id>", methods=["GET"])
def get_model(model_id):
    """returns model .sav file
    
    Returns:
        file: MODEL_DIR/model<model_id>.sav
    """

    try:
        with open("{}/model-{}.sav".format(MODEL_DIR, model_id), "rb") as file:
            return send_file(file, attachment_filename="model-{}.sav".format(model_id))
    except Exception as err:
        return BadRequest(str(err))


@app.route("/api/get_random_text", methods=["GET"])
def get_random_text():
    return dumps(texts.aggregate([{"$sample": {"size": 1}}]))


############
# Frontend #
############
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/texts/<page_number>", methods=["GET"])
def texts_frontend(page_number):
    return render_template(
        "texts.html",
        texts=requests.get(
            f"http://{API_HOST}:{API_PORT}/api/texts/{(int(page_number)-1)*100}&{int(page_number)*100}"
        ).json()[:100],
        current_page=int(page_number),
    )


@app.route("/data-checker", methods=["GET"])
def data_checker():
    this_text = requests.get("http://{}:{}/api/get_random_text".format(API_HOST, API_PORT)).json()[
        0
    ]
    return render_template(
        "data-checker.html",
        text_id=this_text["_id"]["$oid"],
        data=this_text["data"],
        categories=this_text["categories"],
    )


@app.route("/add-data", methods=["GET"])
def add_data():
    return render_template("add_data.html")


@app.route("/change-data", methods=["GET"])
def change_data():
    return render_template(
        "change_data.html",
        default_id="",
        default_data="",
        default_categories="",
        default_vis="hidden",
    )


@app.route("/change-data/<text_id>", methods=["GET"])
def change_data_with_id(text_id):
    this_text = requests.get("http://{}:{}/api/text/{}".format(API_HOST, API_PORT, text_id)).json()
    return render_template(
        "change_data.html",
        default_id=text_id,
        default_data=this_text["data"],
        default_categories=this_text["categories"],
    )


if __name__ == "__main__":
    app.run(debug=True, host=API_HOST, port=API_PORT)
