from flask_restful import Resource
from flask import jsonify


class Categories(Resource):
    def get(self) -> str:
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
