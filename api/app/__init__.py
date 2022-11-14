from flask import Flask
from flask_login import LoginManager
from mongomodels import MongoClient

mongodb = None
uri = "mongodb+srv://draths:Bremen92@cluster0.95mle.mongodb.net/?retryWrites=true&w=majority"
db_name = "Trader"
coll_name = "BankingInfo"

def init_app():

    global mongodb
    app = Flask(__name__)
    mongodb = MongoClient(uri, db_name, coll_name)

    from .dashboard import dashboard_bp

    from api.app.dashboard import dashboard_bp

    app.register_blueprint(dashboard_bp)

    return app


