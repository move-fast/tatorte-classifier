from flask import Flask


def create_app():
    app = Flask("tatorte_api")
    from tatorte_classifier import api
    from tatorte_classifier import frontend

    app.register_blueprint(api.bp)
    app.register_blueprint(frontend.bp)
    return app
