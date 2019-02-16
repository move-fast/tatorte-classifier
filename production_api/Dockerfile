FROM python:3.6

COPY . /app
WORKDIR /app

RUN pip install -r ./requirements.txt
RUN python -c "import nltk; nltk.download('stopwords')"

CMD [ "python", "./api.py" ]
