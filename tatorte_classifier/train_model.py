import os
from typing import Callable, Tuple

import dill
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

from configuration import MODEL_DIR
from tatorte_classifier.model import Model
from tatorte_classifier.preprocess_data import DataPreprocessor


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


def create_model(clf, clf_params, vect_params) -> Model:
    model = Model(clf, clf_params, vect_params)
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
) -> Tuple:
    return (
        accuracy_score(y_train, model(x_train)),
        accuracy_score(y_test, model(x_test)),
        confusion_matrix(y_test, model(x_test)),
    )


def save_model(model, model_id):
    # id = uuid.uuid4().hex[:8]
    print(os.path.join(MODEL_DIR, "model-{}.sav".format(model_id)))
    dill.dump(model, open(os.path.join(MODEL_DIR, "model-{}.sav".format(model_id)), "wb"))
    # return id


def main(x, y, clf, clf_params, vect_params, test_size=0.3, values_per_category=900):
    # print("Loading Data...", end=" ")
    # print("Finished!")
    # print("Preprocessing Data...")
    # print("  Cleaning Data")
    x, y = clean_data(x, y)
    # print("  Balancing Data")
    x, y = balance_data(x, y, values_per_category)
    # print("  Splitting Data")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    # print("Finished!")
    model = create_model(clf, clf_params, vect_params)
    # print("Created Model")
    # print("Training Model...", end=" ")
    model = train_model(x_train, y_train, model)
    # print("Finished!")
    train_acc, test_acc, conf_matrix = evaluate_model(x_train, x_test, y_train, y_test, model)
    # id = save_model(model)
    print("Finished Training")
    # print("Saved Model")
    return model, train_acc, test_acc, conf_matrix

