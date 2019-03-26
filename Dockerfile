FROM python:3.6


COPY . /app

#### Installs ####
RUN pip install -r /app/requirements.txt
RUN python -c "import nltk; nltk.download('stopwords')"

CMD [ "python", "/app/run.py" ]
