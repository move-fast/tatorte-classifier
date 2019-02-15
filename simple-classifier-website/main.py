from flask import Flask, render_template
import numpy as np

import pickle
import pandas as pd

app = Flask(__name__, template_folder="templates")

data = np.load("data.npy")
preds = data[2]
titles = data[0]
descs = data[1]
probs = data[3]
colors = data[4]


@app.route("/")
def hello_world():
    return render_template("index.html", titles=enumerate(zip(titles, preds, probs, colors)))


@app.route("/<idx>")
def get_details(idx):
    # return get_text_color_coded(descs[int(idx)])
    return descs[int(idx)]


if __name__ == "__main__":
    app.run(debug=True)
