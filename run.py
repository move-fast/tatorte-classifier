from tatorte_classifier import create_app
from configuration import ProductionConfig, DevelopementConfig, FLASK_MODE, PORT, HOST

app = create_app()
if FLASK_MODE == "production":
    app.config.from_object(ProductionConfig)
elif FLASK_MODE == "developement":
    app.config.from_object(DevelopementConfig)
app.run(host=HOST, port=PORT)
