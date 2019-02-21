FROM python:3.6


COPY . /

#### Installs ####
RUN pip install -r /requirements.txt
RUN python -c "import nltk; nltk.download('stopwords')"

CMD [ "python", "./app/api.py" ]
