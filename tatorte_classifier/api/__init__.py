from flask import current_app
from flask_restful import Api

from tatorte_classifier.api.texts import Text, TextList
from tatorte_classifier.api.models import (
    ClassifierModel,
    ClassifierModelList,
    ClassifierModelPredict,
    ClassifierModelOptions,
)
from tatorte_classifier.api.categories import Categories

api = Api(current_app)

api.add_resource(Text, "/api/texts/<text_id>/")
api.add_resource(TextList, "/api/texts/")
api.add_resource(Categories, "/api/categories/")
api.add_resource(ClassifierModel, "/api/models/<model_filename>/")
api.add_resource(ClassifierModelList, "/api/models/")
api.add_resource(ClassifierModelPredict, "/api/models/<model_name>/predict/")
api.add_resource(ClassifierModelOptions, "/api/model_options/<model_name>/")
