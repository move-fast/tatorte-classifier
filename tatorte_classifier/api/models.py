import logging
import os
import uuid
from threading import Thread

from pymongo import DESCENDING
from bson.json_util import dumps
from flask import jsonify, request, send_file
from flask_restful import Resource
from werkzeug.exceptions import BadRequest
import numpy as np
import datetime

from configuration import MODEL_DIR
from tatorte_classifier.machine_learning.get_prediction import get_predictions, load_model
from tatorte_classifier.machine_learning.preprocess_data import DataPreprocessor
from tatorte_classifier.machine_learning.train_model import save_model, train_model
from tatorte_classifier.database import get_all_models, get_model, create_model, get_all_texts

logger = logging.getLogger(__name__)


class Model(Resource):
    def get(self, model_id):
        """returns model .sav file

        Returns:
            file: MODEL_DIR/model_id
        """

        try:
            return send_file(
                "{}/{}.sav".format(MODEL_DIR, model_id), attachment_filename=model_id + ".sav"
            )
        except Exception as err:
            logger.error(str(err))
            return BadRequest(str(err))

    def delete(self, model_id):  # TODO: still implement
        pass


class Models(Resource):
    def get(self):
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

        return dumps(get_all_models())

    def post(self):
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
                this_model, train_acc, test_acc, _ = train_model(
                    x, y, **request_json
                )  # train model
                # save model
                this_id = uuid.uuid4().hex[:8]
                save_model(this_model, str(this_id))
                # add model to database
                create_model(
                    "model-{}.sav".format(this_id),
                    {"train_acc": train_acc, "test_acc": test_acc},
                    request_json,
                    "",
                )
                logger.info(f"Trained new model with id: {this_id}")
            except Exception as err:
                logger.error(str(err))
                # insert model with
                create_model("", {}, request_json, str(err))

        try:
            data = list(get_all_texts({"data": 1, "categories": 1}))
            data = np.asarray([[i["data"], i["categories"][0]] for i in data]).T
            x, y = data[0], data[1].astype(int)
            thread = Thread(target=lambda: train(x, y))
            thread.start()
            return "Now training Model"
        except Exception as err:
            logger.error(str(err))
            return BadRequest(str(err))


class ModelPredict(Resource):
    def __init__(self):
        self.preprocessor = DataPreprocessor()

    def post(self, model_id):
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
            desc = self.preprocessor(desc)
            preds = np.asarray(get_predictions([desc], load_model(model_id), max_categories)).T
            preds = [{"category": pred_idx, "probability": pred} for pred_idx, pred in preds]
            return jsonify({"predictions": preds})
        except Exception as err:
            logger.error(str(err))
            return BadRequest(str(err))


class ModelOptions(Resource):
    def get(self, model_name):
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
            logger.info("/api/get_model_options got unexpected Input")
            return BadRequest("please choose one of the classifiers: [svm, sgd, nn]")
