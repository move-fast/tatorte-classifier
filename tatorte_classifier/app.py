import datetime
import os
import uuid
from threading import Thread
import json

import numpy as np
import pymongo
from bson.json_util import dumps
from bson.objectid import ObjectId
from flask import Flask, jsonify, render_template, request, send_file, url_for
from werkzeug.exceptions import BadRequest

from configuration import (
    API_HOST,
    API_PORT,
    CURRENT_MODEL_PATH,
    MODEL_DIR,
    MONGO_PORT,
    MONGO_URL,
    MONGO_AUTH,
    TEMPLATE_FOLDER,
)
from get_prediction import get_predictions, load_model
from preprocess_data import DataPreprocessor
from train_model import main as train_model
from train_model import save_model

# TODO: Change function names
# TODO: Make stuff more consistent

app = Flask("tatorte_api", template_folder=TEMPLATE_FOLDER)
model = load_model()
preprocessor = DataPreprocessor()
client = pymongo.MongoClient(
    os.getenv("MONGODB_URI")
    # "mongodb://{}{}:{}/tatorte-db".format(MONGO_AUTH, MONGO_URL, MONGO_PORT)
)
db = client.get_database()  # ["tatorte-db"]
texts = db["texts"]
texts.create_index([("data", "text")])
models = db["models"]

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


# Data Api
@app.route("/api/texts/start=<start>&end=<end>", methods=["GET"])
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


@app.route("/api/delete_text/<text_id>")
def delete_text(text_id: str):
    """Deletes text document with id == <text_id>
    """

    try:
        texts.delete_one({"_id": ObjectId(text_id)})
        return jsonify({"success": True})
    except Exception as err:
        return BadRequest(str(err))


@app.route("/api/get_texts_count", methods=["GET"])
def get_n_texts():
    """Returns the number of text documents in the database
    
    Returns:
        int -- the number of text documents in the database
    """

    return texts.count()


@app.route("/api/get_random_text", methods=["GET"])
def get_random_text():
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
        Error message or "Now training model". To gvet access to the model just class /api/models and look for your model
    """

    request_json = request.get_json()

    def train(x, y):
        try:
            this_model, train_acc, test_acc, _ = train_model(x, y, **request_json)
            this_id = uuid.uuid4().hex[:8]
            save_model(this_model, str(this_id))
            model_id = models.insert_one(
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
            ).inserted_id
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


@app.route("/api/change_model/<model_name>", methods=["GET"])
def change_model(model_name: str):
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


@app.route("/api/models", methods=["GET"])
def get_models():
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


@app.route("/api/model/<model_name>", methods=["GET"])
def get_model(model_name):
    """returns model .sav file
    
    Returns:
        file: MODEL_DIR/model_name
    """

    try:
        return send_file("{}/{}".format(MODEL_DIR, model_name), attachment_filename=model_name)
    except Exception as err:
        return BadRequest(str(err))


@app.route("/api/get_model_options/<model_name>")
def get_model_options(model_name):
    """Get list of params, which can be passed to different classifiers
    
    Arguments:
        model_name {str} -- The name of the classifier. [sgd, svm, nn]
    
    Returns for Example (sgd):
        {
            "loss": ["log", "modified_huber", "squared_hinge", "perceptron"], # list means that there should be a dropdown selector
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


############
# Frontend #
############
@app.route("/", methods=["GET"])
def index():
    """Home page with link to all sub-pages. Upper half is Data and the other half is Model
    """

    return render_template("index.html")


@app.route("/texts/<page_number>", methods=["GET"])
def texts_frontend(page_number: str):
    """Get texts sorted by modification with pagenumber being results (<page_number> - 1) * 100 to <page_number>*100      
    
    Arguments:
        page_number {int} -- The pagenumber
    
    Returns:
        html
    """

    return render_template(
        "texts.html",
        texts=json.loads(get_texts((int(page_number) - 1) * 100, int(page_number) * 100)),
        current_page=int(page_number),
    )


@app.route("/data-checker", methods=["GET"])
def data_checker():
    """simple page, where you get a randomly selected text-document and you need to annotate it. 
    
    Returns:
        html
    """

    this_text = json.loads(get_random_text())[0]
    return render_template(
        "data-checker.html",
        text_id=this_text["_id"]["$oid"],
        data=this_text["data"],
        categories=this_text["categories"],
    )


@app.route("/add-data", methods=["GET"])
def add_data():
    """Frontend for adding text-documents by providing categories and data/description
    
    Returns:
        html
    """

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
def change_data_with_id(text_id: str):
    """change the categories of a text document given a id
    
    Arguments:
        text_id {str} -- the text_id provided by mongo_db
    
    Returns:
        html
    """

    this_text = json.loads(get_text(text_id))
    return render_template(
        "change_data.html",
        default_id=text_id,
        default_data=this_text["data"],
        default_categories=this_text["categories"],
    )


@app.route("/new-model", methods=["GET"])
def new_model_frontend():
    """Frontend for creating new models
    
    Returns:
        html
    """

    return render_template("new_model.html")


@app.route("/get_predictions", methods=["GET"])
def get_predictions_frontend():
    """Frontend for viewing all trained models and their performances

    Returns:
        html
    """

    return render_template("get_predictions.html", models=json.loads(get_models()))


@app.route("/models", methods=["GET"])
def models_frontend():
    """Frontend for viewing all trained models and their performances

    Returns:
        html
    """

    return render_template("models.html", models=json.loads(get_models()))


if __name__ == "__main__":
    app.run(debug=True, host=API_HOST, port=API_PORT)
