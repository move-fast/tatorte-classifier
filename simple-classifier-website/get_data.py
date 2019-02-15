import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re

stemmer = SnowballStemmer("german", ignore_stopwords=True)

analyzer = CountVectorizer().build_analyzer()


def stemmed_words(doc):
    return (stemmer.stem(w) for w in analyzer(doc))


data = pd.read_csv("/home/peer/Data/tatorte.csv", ";")
data = data[data.description.str.len() > 40]
data = data[["title", "description"]].dropna().to_numpy()[:1000].astype(str)
# remove contact data
remove_emails = np.vectorize(lambda x: re.sub("\S*@\S*\s?", "", re.sub("email:", "", re.sub("e-mail:", "", x.lower()))))
remove_telephones = np.vectorize(
    lambda x: re.sub(
        "(\(?([\d \-\)\–\+\/\(]+)\)?([ .-–\/]?)([\d]+))", "", re.sub("tel.:", "", re.sub("telefon:", "", x.lower()))
    )
)
remove_links = np.vectorize(
    lambda x: re.sub(
        "http://", "", re.sub("https://", "", re.sub(r"www.[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*", "", x.lower()))
    )
)
x = remove_links(remove_telephones(remove_emails(data.T[1])))
print("loaded data")
model = joblib.load("model.joblib")
print("loaded model")
pred = model.predict_proba(x)
pred_idx = np.argmax(pred, 1)
probs = pred[np.arange(len(pred_idx)), pred_idx]
colors = np.full((1000), "#ffffff")
colors[np.where(pred_idx == 0)[0]] = "#ff8888"
colors[np.where(pred_idx == 1)[0]] = "#88ff88"
colors[np.where(pred_idx == 2)[0]] = "#8888ff"
colors[np.where(probs < 0.6)[0]] = "#ffffff"
data = np.concatenate((data.T, pred_idx.reshape(1, -1), probs.reshape(1, -1), colors.reshape(1, -1)))
np.save("data.npy", data)
