from model import Model
from preprocess_data import DataPreprocessor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import dill
import numpy as np
import time
import os
from configuration import MODEL_DIR


def load_data():
    x = np.load("/home/peer/Data/data_x.npy")
    y = np.load("/home/peer/Data/data_y.npy")
    return x, y


def clean_data(x, y):
    y[np.where(y == 10)[0]] = 7
    y[np.where(y == 8)[0]] = 7
    x = x[np.where(y != 7)]
    y = y[np.where(y != 7)]
    y[np.where(y == 2)[0]] = 90
    y[np.where(y == 3)[0]] = 2
    y[np.where(y == 4)[0]] = 2
    y[np.where(y == 5)[0]] = 2
    y[np.where(y == 9)[0]] = 4
    y[np.where(y == 90)[0]] = 3
    y[np.where(y == 6)[0]] = 5

    # automate Data cleaning
    str_contain = np.vectorize(lambda x: "verkehrskontroll" in x.lower())
    y[np.intersect1d(np.where(str_contain(x))[0], np.where(y == 4)[0])] = 5

    str_contain = np.vectorize(lambda x: "eingebroch" in x.lower())
    y[np.intersect1d(np.where(str_contain(x))[0], np.where(y == 1)[0])] = 2

    str_contain = np.vectorize(lambda x: "alkohol" in x.lower())
    idxs = np.where(str_contain(x))
    y = np.delete(y, idxs)
    x = np.delete(x, idxs)

    str_contain = np.vectorize(lambda x: "dienstagmorg" in x.lower())
    idxs = np.intersect1d(np.where(str_contain(x)), np.where(y == 2))
    y = np.delete(y, idxs)
    x = np.delete(x, idxs)

    str_contain = np.vectorize(lambda x: "fahrrad" in x.lower())
    idxs = np.intersect1d(np.where(str_contain(x))[0], np.where(y == 2)[0])
    x = np.delete(x, idxs)
    y = np.delete(y, idxs)

    preprocessor = DataPreprocessor()
    preprocessor = np.vectorize(preprocessor)
    x = preprocessor(x)
    return x, y


def balance_data(x, y, n_per_class):
    for i in np.unique(y):
        idxs = np.where(y == i)[0]
        np.random.shuffle(idxs)
        idxs = idxs[: (len(idxs) - n_per_class)]
        x = np.delete(x, idxs)
        y = np.delete(y, idxs)
    return x, y


def create_model():
    model = Model(
        {"alpha": 1e-5, "max_iter": 100, "loss": "log", "penalty": "l2"}, {"ngram_range": (1, 4)}
    )
    return model


def train_model(x_train, x_test, y_train, y_test, model):
    model.pipeline = model.pipeline.fit(x_train, y_train)
    print("Train acc:", accuracy_score(y_train, model(x_train)))
    print("Test acc:", accuracy_score(y_test, model(x_test)))
    return model


def save_model(model):
    dill.dump(model, open(os.path.join(MODEL_DIR, "model-{}.sav".format(time.time())), "wb"))


def main():
    print("Loading Data...", end=" ")
    x, y = load_data()
    print("Finished!")
    print("Preprocessing Data...")
    print("  Cleaning Data")
    x, y = clean_data(x, y)
    print("  Balancing Data")
    x, y = balance_data(x, y, 900)
    print("  Splitting Data")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    print("Finished!")
    model = create_model()
    print("Created Model")
    print("Training Model...", end=" ")
    model = train_model(x_train, x_test, y_train, y_test, model)
    print("Finished!")
    save_model(model)
    print("Saved Model")
    return model


if __name__ == "__main__":
    main()
