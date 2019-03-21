from flask import current_app
from flask_restful import Api

from tatorte_classifier.api.texts import Text, Texts, TextsStartEnd
from tatorte_classifier.api.models import Model, Models, ModelPredict, ModelOptions
from tatorte_classifier.api.categories import Categories

api = Api(current_app)

api.add_resource(Text, "/api/texts/<text_id>")
api.add_resource(Texts, "/api/texts")
api.add_resource(TextsStartEnd, "/api/texts/start=<int:start>&end=<int:end>")
api.add_resource(Categories, "/api/categories")
api.add_resource(Model, "/api/models/<model_filename>")
api.add_resource(Models, "/api/models")
api.add_resource(ModelPredict, "/api/models/<model_id>/predict")
api.add_resource(ModelOptions, "/api/model_options/<model_name>")
