from flask import Blueprint, render_template
from configuration import TEMPLATE_FOLDER
import json
from tatorte_classifier.views.api import get_texts, get_models, get_random_text, get_text

bp = Blueprint("frontend", __name__, template_folder=TEMPLATE_FOLDER, url_prefix="/")


@bp.route("/", methods=["GET"])
def index():
    """Home page with link to all sub-pages. Upper half is Data and the other half is Model
    """

    return render_template("index.html")


@bp.route("/texts/<page_number>", methods=["GET"])
def texts_frontend(page_number: str):
    """Get texts sorted by modification with pagenumber being results (<page_number> - 1) * 100 to <page_number>*100      
    
    Arguments:
        page_number {int} -- The pagenumber
    
    Returns:
        html
    """

    return render_template(
        "texts.html",
        texts=json.loads(get_texts((int(page_number) - 1) * 100, int(page_number) * 100)),
        current_page=int(page_number),
    )


@bp.route("/data-checker", methods=["GET"])
def data_checker():
    """simple page, where you get a randomly selected text-document and you need to annotate it. 
    
    Returns:
        html
    """

    this_text = json.loads(get_random_text())[0]
    return render_template(
        "data-checker.html",
        text_id=this_text["_id"]["$oid"],
        data=this_text["data"],
        categories=this_text["categories"],
    )


@bp.route("/add-data", methods=["GET"])
def add_data():
    """Frontend for adding text-documents by providing categories and data/description
    
    Returns:
        html
    """

    return render_template("add_data.html")


@bp.route("/change-data", methods=["GET"])
def change_data():
    return render_template(
        "change_data.html",
        default_id="",
        default_data="",
        default_categories="",
        default_vis="hidden",
    )


@bp.route("/change-data/<text_id>", methods=["GET"])
def change_data_with_id(text_id: str):
    """change the categories of a text document given a id
    
    Arguments:
        text_id {str} -- the text_id provided by mongo_db
    
    Returns:
        html
    """
    this_text = get_text(text_id)
    if isinstance(this_text, str):
        return "There is no text with that id"

    this_text = json.loads(get_text(text_id))
    return render_template(
        "change_data.html",
        default_id=text_id,
        default_data=this_text["data"],
        default_categories=this_text["categories"],
    )


@bp.route("/new-model", methods=["GET"])
def new_model_frontend():
    """Frontend for creating new models
    
    Returns:
        html
    """

    return render_template("new_model.html")


@bp.route("/get_predictions", methods=["GET"])
def get_predictions_frontend():
    """Frontend for viewing all trained models and their performances

    Returns:
        html
    """

    return render_template("get_predictions.html", models=json.loads(get_models()))


@bp.route("/models", methods=["GET"])
def models_frontend():
    """Frontend for viewing all trained models and their performances

    Returns:
        html
    """

    return render_template("models.html", models=json.loads(get_models()))

