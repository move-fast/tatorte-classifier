import numpy as np
import requests
import tqdm

"""
This tool imports data_x.npy and data_y.npy generated by create_data.ipynb into the mongo db

"""


def convert_to_string(x):
    try:
        return x.decode("utf-8")
    except:
        return x


X = np.load("/home/peer/Data/data_x.npy")
X = np.vectorize(convert_to_string)(X)
Y = np.load("/home/peer/Data/data_y.npy")

X = X[np.intersect1d(np.where(Y != 7), np.intersect1d(np.where(Y != 8), np.where(Y != 10)))]
Y = Y[np.intersect1d(np.where(Y != 7), np.intersect1d(np.where(Y != 8), np.where(Y != 10)))]
# combine categories 3, 4, 5
Y[np.where(Y == 4)[0]] = 3
Y[np.where(Y == 5)[0]] = 3
# change indexes
Y[np.where(Y == 6)[0]] = 4
Y[np.where(Y == 9)[0]] = 5
Y -= 1

data = [{"data": x, "categories": [int(y)]} for x, y in zip(X, Y)]
for i in tqdm.tqdm(data[4000:8000]):
    requests.post("http://localhost:5000/api/texts", json=i)
