import pymongo
import numpy as np
import datetime
from configuration import MONGO_PASSWORD, MONGO_PORT, MONGO_URL, MONGO_USER

client = pymongo.MongoClient(
    "mongodb://{}:{}@{}:{}/tatorte-db".format(MONGO_USER, MONGO_PASSWORD, MONGO_URL, MONGO_PORT)
)
db = client["tatorte-db"]
texts = db["texts"]


def convert_to_string(x):
    try:
        return x.decode("utf-8")
    except:
        return x


X = np.load("./data/data_x.npy")
X = np.vectorize(convert_to_string)(X)
Y = np.load("./data/data_y.npy")

X = X[np.intersect1d(np.where(Y != 7), np.intersect1d(np.where(Y != 8), np.where(Y != 10)))]
Y = Y[np.intersect1d(np.where(Y != 7), np.intersect1d(np.where(Y != 8), np.where(Y != 10)))]
# combine categories 3, 4, 5
Y[np.where(Y == 4)[0]] = 3
Y[np.where(Y == 5)[0]] = 3
# change indexes
Y[np.where(Y == 6)[0]] = 4
Y[np.where(Y == 9)[0]] = 5
Y -= 1

current_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

data = [
    {"data": x, "categories": [int(y)], "time_created": current_date, "time_modified": current_date}
    for x, y in zip(X, Y)
]
print(len(data))
for i in range(1, len(data) // 1000 + 2):
    print(i)
    texts.insert_many(data[(i - 1) * 1000 : i * 1000])
