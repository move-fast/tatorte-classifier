import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.externals import joblib

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer


class Model:
    def __init__(self, clf_params: dict, vect_params: dict):
        self.stemmer = self._build_stemmer()
        self.pipeline = Pipeline(
            [
                (
                    "vect",
                    TfidfVectorizer(
                        stop_words=set(stopwords.words("german")),
                        analyzer=self.stemmer,
                        **vect_params
                    ),
                ),
                ("clf", SGDClassifier(**clf_params)),
            ]
        )

    def __call__(self, x):
        return self.pipeline.predict(x)

    def _build_stemmer(self):
        stemmer = SnowballStemmer("german", ignore_stopwords=True)
        analyzer = TfidfVectorizer().build_analyzer()

        def stemmed_words(doc):
            return (stemmer.stem(w) for w in analyzer(doc))

        return stemmed_words

