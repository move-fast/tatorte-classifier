import datetime
import os
import uuid
from threading import Thread
from typing import Union


import numpy as np
import pymongo
from bson.json_util import dumps
from bson.objectid import ObjectId
from flask import Blueprint, jsonify, request, send_file
from werkzeug.exceptions import BadRequest

from configuration import CURRENT_MODEL_PATH, MODEL_DIR, MONGODB_URI
from tatorte_classifier.get_prediction import get_predictions, load_model
from tatorte_classifier.model import Model
from tatorte_classifier.preprocess_data import DataPreprocessor
from tatorte_classifier.train_model import main as train_model
from tatorte_classifier.train_model import save_model

bp = Blueprint("api", __name__, url_prefix="/api")
try:
    model = load_model()
except:  # If their is no model at the moment
    model = Model("sgd", {}, {})
preprocessor = DataPreprocessor()
client = pymongo.MongoClient(MONGODB_URI)
db = client.get_database()  # ["tatorte-db"]
texts = db["texts"]
texts.create_index([("data", "text")])
models = db["models"]


@bp.route("/get_prediction", methods=["POST"])
def get_preds() -> Union[str, BadRequest]:
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


@bp.route("/categories", methods=["GET"])
def get_key() -> str:
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


# Data Api
@bp.route("/texts/start=<int:start>&end=<int:end>", methods=["GET"])
def get_texts(start: int, end: int) -> Union[str, BadRequest]:
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
    try:
        return dumps(texts.find().sort("time_modified", pymongo.DESCENDING)[start:end])
    except Exception as err:
        return BadRequest(str(err))


@bp.route("/text/<text_id>", methods=["GET"])
def get_text(text_id: str) -> Union[str, BadRequest]:
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


@bp.route("/create_text", methods=["POST"])
def create_text() -> Union[str, BadRequest]:
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


@bp.route("/change_categories", methods=["POST"])
def change_text() -> Union[str, BadRequest]:
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
        text_id = request_json["id"]
        texts.update_one(
            {"_id": ObjectId(text_id)},
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


@bp.route("/delete_text/<text_id>")
def delete_text(text_id: str) -> Union[str, BadRequest]:
    """Deletes text document with id == <text_id>
    """

    try:
        texts.delete_one({"_id": ObjectId(text_id)})
        return jsonify({"success": True})
    except Exception as err:
        return BadRequest(str(err))


@bp.route("/get_texts_count", methods=["GET"])
def get_n_texts() -> str:
    """Returns the number of text documents in the database

    Returns:
        int -- the number of text documents in the database
    """

    return texts.count()


@bp.route("/get_random_text", methods=["GET"])
def get_random_text() -> str:
    """Gets a randomly selected text out of the text database

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

    return dumps(texts.aggregate([{"$sample": {"size": 1}}]))


# Model API
@bp.route("/train_model", methods=["POST"])
def new_model() -> Union[str, BadRequest]:
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
        Error message or "Now training model". To get access to the model just call /api/models
        and look for your model or navigate on the gui to the "models" sections
    """

    request_json = request.get_json()

    def train(x: np.ndarray, y: np.ndarray) -> None:
        try:
            this_model, train_acc, test_acc, _ = train_model(x, y, **request_json)
            this_id = uuid.uuid4().hex[:8]
            save_model(this_model, str(this_id))
            models.insert_one(
                {
                    "time_created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "model_url": "model-{}.sav".format(this_id),
                    "performance_data": {"train_acc": train_acc, "test_acc": test_acc},
                    "metadata": {
                        "clf": request_json["clf"],
                        "clf_params": request_json["clf_params"],
                        "vect_params": request_json["vect_params"],
                        "test_size": request_json["test_size"],
                        "values_per_category": request_json["values_per_category"],
                    },
                    "error_message": "",
                }
            )
        except Exception as err:
            models.insert_one(
                {
                    "time_created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "model_url": "",
                    "error_message": str(err),
                    "performance_data": {},
                    "metadata": {
                        "clf": request_json["clf"],
                        "clf_params": request_json["clf_params"],
                        "vect_params": request_json["vect_params"],
                    },
                }
            )

    try:
        data = list(texts.find({}, {"data": 1, "categories": 1}))
        data = np.asarray([[i["data"], i["categories"][0]] for i in data]).T
        x, y = data[0], data[1].astype(int)
        thread = Thread(target=lambda: train(x, y))
        thread.start()
        return "Now training Model"
    except Exception as err:
        return BadRequest(str(err))


@bp.route("/change_model/<model_name>", methods=["GET"])
def change_model(model_name: str) -> Union[str, BadRequest]:
    """changes the current model, which is used for /api/get_predictions

    Arguments:
        model_name {str} -- the name of the model (model-<id>.sav)

    Returns:
        error message or success message
    """

    global model
    try:
        try:  # only if VURRENT_MODEL exists
            os.system("rm -f " + CURRENT_MODEL_PATH)
        except:
            pass
        os.system("cp {}/{} {}".format(MODEL_DIR, model_name, CURRENT_MODEL_PATH))
        model = load_model()
        return jsonify({"success": True})
    except Exception as err:
        return BadRequest(str(err))


@bp.route("/models", methods=["GET"])
def get_models() -> str:
    """Get a list of metadata, performance-data for all the models

    Returns:
        [
            {
                "_id": {
                    "$oid": "5c6ec0288c55ec0013a2bd81"
                },
                "time_created": "2019-02-21 15:13:44",
                "model_url": "model-8d5e4706.sav",
                "performance_data": {
                    "train_acc": 1,
                    "test_acc": 0.7123287671232876
                },
                "metadata": {
                    "clf": "sgd",
                    "clf_params": {
                        "alpha": 0.0001,
                        "loss": "log",
                        "max_iter": 100,
                        "penalty": "l2"
                    },
                    "vect_params": {
                        "ngram_range": [
                            1,
                            2
                        ]
                    },
                    "test_size": 0.3,
                    "values_per_category": 200
                },
                "error_message": ""
            }, ...
        ]
    """

    return dumps(models.find().sort("time_created", pymongo.DESCENDING))


@bp.route("/model/<model_name>", methods=["GET"])
def get_model(model_name):
    """returns model .sav file

    Returns:
        file: MODEL_DIR/model_name
    """

    try:
        return send_file("{}/{}".format(MODEL_DIR, model_name), attachment_filename=model_name)
    except Exception as err:
        return BadRequest(str(err))


@bp.route("/get_model_options/<model_name>")
def get_model_options(model_name: str) -> Union[str, BadRequest]:
    """Get list of params, which can be passed to different classifiers

    Arguments:
        model_name {str} -- The name of the classifier. [sgd, svm, nn]

    Returns for Example (sgd):
        {
            "loss": ["log", "modified_huber", "squared_hinge", "perceptron"], # list means, that there should be a dropdown selector
            "penalty": ["l2", "l1", "elastic_net"],
            "alpha": 0.0001, # number means that there should be a number input
            "max_iter": 100,
        }
    """

    if model_name == "sgd":
        return jsonify(
            {
                "loss": ["log", "modified_huber", "squared_hinge", "perceptron"],
                "penalty": ["l2", "l1", "elastic_net"],
                "alpha": 0.0001,
                "max_iter": 100,
            }
        )
    elif model_name == "svm":
        return jsonify(
            {
                "C": 1.0,
                "kernel": ["linear", "poly", "rbf", "sigmoid", "precomputed"],
                "degree": 3,
                "probability": False,
                "tol": 0.001,
                "max_iter": -1,
            }
        )
    elif model_name == "nn":
        return jsonify(
            {
                "hidden_layer_sizes": "50 50",
                "activation": ["identity", "logistic", "tanh", "relu"],
                "solver": ["adam", "lbfgs", "sgd"],
                "alpha": 0.0001,
                "learning_rate": ["constant", "invscaling", "adaptive"],
                "max_iter": 200,
            }
        )
    else:
        return BadRequest("please choose one of the classifiers: [svm, sgd, nn]")
