from flask import Flask
from flask_cors import *
def register_blueprint(app):
    from app.api.Face import Face
    app.register_blueprint(Face)
app= Flask(__name__)


def create_app():

    app.config.from_object('app.config.setting')
    app.config.from_object('app.config.secure')
    register_blueprint(app)
    CORS(app, supports_credentials=True)
    return app
