from model import Model
from preprocess_data import DataPreprocessor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import dill
import numpy as np
import time
import os
from configuration import MODEL_DIR, DATA_X_PATH, DATA_Y_PATH
from typing import Callable, Tuple


def load_data():
    x = np.load(DATA_X_PATH)
    y = np.load(DATA_Y_PATH)
    return x, y


def _change_category(
    x: np.ndarray, y: np.ndarray, old_category: int, new_category: int, condition: Callable
) -> np.ndarray:
    y[np.intersect1d(np.where(condition(x))[0], np.where(y == old_category)[0])] = new_category
    return y


def _drop_all_containing_keyword(
    x: np.ndarray, y: np.ndarray, keyword: str, category: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    str_contain = np.vectorize(lambda x: keyword in x.lower())
    idxs = np.where(str_contain(x))
    if category != None:
        idxs = np.intersect1d(idxs, np.where(y == category))
    y = np.delete(y, idxs)
    x = np.delete(x, idxs)
    return x, y


def clean_data(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Key:
    #   1 - Feuer
    #   2 - Mord
    #   3 - Überfall/Körperverletzung
    #   4 - Unfall
    #   5 - Drogen
    # Delete "dirty" categories 7, 8, 10
    x = x[np.intersect1d(np.where(y != 7), np.intersect1d(np.where(y != 8), np.where(y != 10)))]
    y = y[np.intersect1d(np.where(y != 7), np.intersect1d(np.where(y != 8), np.where(y != 10)))]
    # combine categories 3, 4, 5
    y[np.where(y == 4)[0]] = 3
    y[np.where(y == 5)[0]] = 3
    # change indexes
    y[np.where(y == 6)[0]] = 4
    y[np.where(y == 9)[0]] = 5

    # automate Data cleaning
    str_contain = np.vectorize(lambda x: "verkehrskontroll" in x.lower())
    y = _change_category(x, y, 5, 4, str_contain)

    str_contain = np.vectorize(lambda x: "eingebroch" in x.lower())
    y = _change_category(x, y, 4, 3, str_contain)

    x, y = _drop_all_containing_keyword(x, y, "alkohol")

    x, y = _drop_all_containing_keyword(x, y, "dienstagmorg", category=3)

    x, y = _drop_all_containing_keyword(x, y, "fahrrad", category=3)

    preprocessor = DataPreprocessor()
    preprocessor = np.vectorize(preprocessor)
    x = preprocessor(x)
    return x, y


def balance_data(x: np.ndarray, y: np.ndarray, n_per_class: int) -> Tuple[np.ndarray, np.ndarray]:
    for i in np.unique(y):
        idxs = np.where(y == i)[0]
        np.random.shuffle(idxs)
        idxs = idxs[: (len(idxs) - n_per_class)]
        x = np.delete(x, idxs)
        y = np.delete(y, idxs)
    return x, y


def create_model(clf_params, vect_params) -> Model:
    model = Model("sgd", clf_params, vect_params)
    return model


def _print_top10_features(vectorizer, clf, class_labels):
    """Prints features with the highest coefficient values, per class"""
    feature_names = vectorizer.get_feature_names()
    for i, class_label in enumerate(class_labels):
        top10 = np.argsort(clf.coef_[i])[-10:]
        print("%s: %s" % (class_label, " ".join(feature_names[j] for j in top10)))


def train_model(x_train: np.ndarray, y_train: np.ndarray, model: Model) -> Model:
    model.pipeline = model.pipeline.fit(x_train, y_train)
    return model


def evaluate_model(
    x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, model: Model
) -> None:
    print("Train acc:", accuracy_score(y_train, model(x_train)))
    print("Test acc:", accuracy_score(y_test, model(x_test)))
    print("Confusion Matrix:\n", confusion_matrix(y_test, model(x_test)))
    _print_top10_features(
        model.pipeline.named_steps["vect"],
        model.pipeline.named_steps["clf"],
        ["Feuer", "Mord", "Überfall/Körperverletzung", "Unfall", "Drogen"],
    )


def save_model(model):
    dill.dump(model, open(os.path.join(MODEL_DIR, "model-{}.sav".format(time.time())), "wb"))


def main(clf_params, vect_params):
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
    model = create_model(clf_params, vect_params)
    print("Created Model")
    print("Training Model...", end=" ")
    model = train_model(x_train, y_train, model)
    print("Finished!")
    evaluate_model(x_train, x_test, y_train, y_test, model)
    save_model(model)
    print("Saved Model")
    return model


if __name__ == "__main__":
    main({"alpha": 1e-6, "max_iter": 100, "loss": "log", "penalty": "l2"}, {"ngram_range": (1, 4)})
