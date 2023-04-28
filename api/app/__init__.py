from flask import Flask
from flask_login import LoginManager
from mongomodels import MongoClient

from dotenv import dotenv_values
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
config = dotenv_values(BASE_DIR/ ".env")

mongodb = None
uri = config['MONGO_DB_URI']
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


