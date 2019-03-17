import datetime
import os
import uuid
import logging

import pymongo
from flask import current_app
from flask_restful import Api

from configuration import MONGODB_URI
from tatorte_classifier.api.texts import Text, Texts
from tatorte_classifier.api.models import Model, Models, ModelPredict, ModelOptions
from tatorte_classifier.api.categories import Categories

api = Api(current_app, "/api")

logger = logging.getLogger(__name__)

api.add_resource(Text, "/texts/<text_id>")
api.add_resource(Texts, "/texts/", "/texts/start=<int:start>&end=<int:end>")
api.add_resource(Categories, "/categories")
api.add_resource(Model, "/models/<model_id>")
api.add_resource(Models, "/models")
api.add_resource(ModelPredict, "/models/<model_id>/predict")
api.add_resource(ModelOptions, "/model_options/<model_name>")
