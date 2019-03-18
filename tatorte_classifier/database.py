import time

import pymongo
from bson.objectid import ObjectId

from configuration import MONGODB_URI

client = pymongo.MongoClient(MONGODB_URI)
db = client.get_database()
texts = db["texts"]
models = db["models"]


def get_all_models():
    return models.find().sort("time_created", pymongo.DESCENDING)


def get_model(text_id):
    return texts.find_one({"_id": ObjectId(text_id)})


def create_model(model_url, performance_data, metadata, error_message):
    models.insert_one(
        {
            "model_url": model_url,
            "performance_data": performance_data,
            "metadata": metadata,
            "error_message": error_message,
            "time_created": time.time(),
        }
    )


def get_all_texts(projection=None):
    if projection:
        return texts.find({}, projection).sort("time_modified", pymongo.DESCENDING)
    return texts.find().sort("time_modified", pymongo.DESCENDING)


def get_text(text_id):
    return texts.find_one({"_id": ObjectId(text_id)})


def create_text(data, categories):
    texts.insert_one(
        {
            "data": data,
            "categories": categories,
            "time_created": time.time(),
            "time_modified": time.time(),
        }
    )


def modify_text(text_id, categories):
    texts.update_one(
        {"_id": ObjectId(text_id)},
        {"$set": {"categories": categories, "time_modified": time.time()}},
        upsert=False,
    )


def delete_text(text_id):
    texts.delete_one({"_id": ObjectId(text_id)})


def get_random_text():
    return texts.aggregate([{"$sample": {"size": 1}}])
