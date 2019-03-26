from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer


class Model:
    def __init__(self, clf: str, clf_params: dict, vect_params: dict, scale_params: dict = {}):
        """[summary]
 
        Arguments:
            clf {str} -- The Classifier. Options are sgd, nn, svm
            clf_params {dict} -- The params for the classifier
            vect_params {dict} -- The params for the TfidfVectorizer
        """

        self.stemmer = self._build_stemmer()
        self.pipeline = self._build_pipeline(clf, clf_params, vect_params, scale_params)

    def __call__(self, x):
        return self.pipeline.predict(x)

    def _build_stemmer(self):
        stemmer = SnowballStemmer("german", ignore_stopwords=True)
        analyzer = TfidfVectorizer().build_analyzer()

        def stemmed_words(doc):
            return (stemmer.stem(w) for w in analyzer(doc))

        return stemmed_words

    def _build_pipeline(self, clf, clf_params, vect_params, scale_params):
        pipeline_steps = [
            (
                "vect",
                TfidfVectorizer(
                    stop_words=set(stopwords.words("german")), analyzer=self.stemmer, **vect_params
                ),
            )
        ]
        if clf == "sgd":
            from sklearn.linear_model import SGDClassifier

            pipeline_steps.append(("clf", SGDClassifier(**clf_params)))
        elif clf == "svm":
            from sklearn.svm import SVC
            from sklearn.preprocessing import StandardScaler

            pipeline_steps.append(("scale", StandardScaler(**scale_params, with_mean=False)))
            pipeline_steps.append(("clf", SVC(**clf_params)))
        elif clf == "nn":
            from sklearn.neural_network import MLPClassifier

            pipeline_steps.append(("clf", MLPClassifier(**clf_params)))
        pipeline = Pipeline(pipeline_steps)
        return pipeline
