import json

from flask import Blueprint, render_template

from configuration import TEMPLATE_FOLDER, STATIC_FOLDER

# from tatorte_classifier.api import get_models, get_random_text, get_text, get_texts
from tatorte_classifier.database import get_all_texts, get_all_models, get_random_text, get_text

bp = Blueprint(
    "frontend",
    __name__,
    template_folder=TEMPLATE_FOLDER,
    url_prefix="/",
    static_folder=STATIC_FOLDER,
)


@bp.route("/", methods=["GET"])
def index() -> str:
    """Home page with link to all sub-pages. Upper half is Data and the other half is Model
    """

    return render_template("index.html")


@bp.route("/texts/<int:page_number>", methods=["GET"])
def texts_frontend(page_number: int) -> str:
    """Get texts sorted by modification with pagenumber being results (<page_number> - 1) * 100 to <page_number>*100

    Arguments:
        page_number {int} -- The pagenumber

    Returns:
        html
    """
    return render_template(
        "texts.html",
        texts=list(get_all_texts()[(page_number - 1) * 100 : page_number * 100]),
        current_page=int(page_number),
    )


@bp.route("/data-checker", methods=["GET"])
def data_checker() -> str:
    """simple page, where you get a randomly selected text-document and you need to annotate it.

    Returns:
        html
    """

    text = list(get_random_text())[0]
    return render_template(
        "data-checker.html", text_id=text["_id"], data=text["data"], categories=text["categories"]
    )


@bp.route("/add-data", methods=["GET"])
def add_data() -> str:
    """Frontend for adding text-documents by providing categories and data/description

    Returns:
        html
    """

    return render_template("add_data.html")


@bp.route("/change-data", methods=["GET"])
def change_data() -> str:
    """Frontend for changing data. Uses Id for indexing and provides options to change categories
    """

    return render_template(
        "change_data.html",
        default_id="",
        default_data="",
        default_categories="",
        default_vis="hidden",
    )


@bp.route("/change-data/<text_id>", methods=["GET"])
def change_data_with_id(text_id: str) -> str:
    """change the categories of a text document given a id

    Arguments:
        text_id {str} -- the text_id provided by mongo_db

    Returns:
        html
    """
    text = get_text(text_id)
    return render_template(
        "change_data.html",
        default_id=text_id,
        default_data=text["data"],
        default_categories=text["categories"],
    )


@bp.route("/new-model", methods=["GET"])
def new_model_frontend() -> str:
    """Frontend for creating new models

    Returns:
        html
    """

    return render_template("new_model.html")


@bp.route("/predictions/<model_id>", methods=["GET"])
def get_predictions_frontend(model_id) -> str:
    """Frontend for viewing all trained models and their performances

    Returns:
        html
    """

    return render_template("get_predictions.html", model_id=model_id)


@bp.route("/models", methods=["GET"])
def models_frontend() -> str:
    """Frontend for viewing all trained models and their performances

    Returns:
        html
    """

    return render_template("models.html", models=list(get_all_models()))
