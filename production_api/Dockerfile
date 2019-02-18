FROM python:3.6

COPY . /app
WORKDIR /app

#### Constants ####
ENV MODEL_DIR /app/models
ENV CURRENT_MODEL_PATH /app/models/model.sav
# The minimum length of a description for it to be classified
ENV MIN_DESC_LEN 40
# The minimum probability of a description for it to be classified
ENV MIN_PREDICTING_PROBA 0.5
ENV API_HOST 0.0.0.0
ENV API_PORT 5000

#### Installs ####
RUN pip install -r ./requirements.txt
RUN python -c "import nltk; nltk.download('stopwords')"

CMD [ "python", "./api.py" ]
