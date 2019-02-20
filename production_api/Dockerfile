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
ENV MONGO_USER test_user
ENV MONGO_PASSWORD Y76K9p-#wkFa]X%f%g9w
ENV MONGO_URL ds161024.mlab.com
ENV MONGO_PORT 61024
ENV TEMPLATE_FOLDER /app/templates/
#### Installs ####
RUN pip install -r /requirements.txt
RUN python -c "import nltk; nltk.download('stopwords')"

CMD [ "python", "./app/api.py" ]
