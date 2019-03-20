import logging

from flask import jsonify, request
from flask_restful import Resource
from bson.json_util import dumps

from werkzeug.exceptions import BadRequest

from tatorte_classifier.database import (
    get_all_texts,
    get_text,
    modify_text,
    create_text,
    get_random_text,
    delete_text,
)

logger = logging.getLogger(__name__)


class Text(Resource):
    def get(self, text_id):
        """
        Input:
            text_id: [id of text, 'random']
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
            if text_id == "random":
                return dumps(get_random_text())
            return dumps(get_text(text_id))
        except Exception as err:
            logger.error(str(err))
            return BadRequest(str(err))

    def delete(self, text_id):
        """Deletes text document with id == <text_id>
        """

        try:
            delete_text(text_id)
            return jsonify({"success": True})
        except Exception as err:
            logger.error(str(err))
            return BadRequest(str(err))

    def patch(self, text_id):
        """
        Input:
            {
                "categories": [3, 2],
            }

        Returns:
            <id>
        """
        try:
            request_json = request.get_json()
            modify_text(text_id, request_json["categories"])
            return jsonify({"success": True})
        except Exception as err:
            logger.error(str(err))
            return BadRequest(str(err))


class Texts(Resource):
    def get(self):
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
            return dumps(get_all_texts())
        except Exception as err:
            logger.error(str(err))
            return BadRequest(str(err))

    def post(self):
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
            create_text(request_json["data"], request_json["categories"])
            return jsonify({"success": True})
        except Exception as err:
            logger.error(str(err))
            return BadRequest(str(err))


class TextsStartEnd(Resource):
    def get(self, start, end):
        """Get all texts sorted by time modified from starting index to ending index.
 
        Arguments:
            start {int} -- The starting index
            end {int} -- The ending index
 
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

        return dumps(get_all_texts()[start:end])
