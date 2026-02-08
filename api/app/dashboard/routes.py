import json
import pandas as pd

from alpaca_folder.alpaca import positions

from api.app.dashboard import dashboard_bp
from api.app import mongodb
from flask import render_template, jsonify, request, current_app
import plotly_models

@dashboard_bp.route("/pdf", methods = ["GET", "POST"])
def dashboard():

    query = None
    data = mongodb.find_one(query)
    del data["_id"]

    ptf = portfolio()

    return render_template("index.html")

def portfolio():

    ptf = positions()

    return ptf