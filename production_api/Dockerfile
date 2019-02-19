FROM python:3.6


COPY . /

#### Constants ####
ENV MODEL_DIR /models
ENV CURRENT_MODEL_PATH /models/model.sav
# The minimum length of a description for it to be classified
ENV MIN_DESC_LEN 40
# The minimum probability of a description for it to be classified
ENV MIN_PREDICTING_PROBA 0.5
ENV API_HOST 0.0.0.0
ENV API_PORT 5000
ENV DATA_X_PATH /data/data_x.npy
ENV DATA_Y_PATH /data/data_y.npy

#### Installs ####
RUN pip install -r /requirements.txt
RUN python -c "import nltk; nltk.download('stopwords')"

CMD [ "python", "./app/api.py" ]
