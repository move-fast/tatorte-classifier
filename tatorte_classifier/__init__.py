from flask import Flask
from configuration import STATIC_FOLDER


def create_app():
    app = Flask("tatorte_api", static_folder=STATIC_FOLDER)
    with app.app_context():  # To be able to use current_app
        from tatorte_classifier import api
    from tatorte_classifier import frontend

    app.register_blueprint(frontend.bp)
    return app
