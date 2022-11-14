import json

import plotly.utils

from api.app.dashboard import dashboard_bp
from api.app import mongodb
from flask_login import current_user
from flask import render_template, jsonify, request, current_app
import plotly_models

@dashboard_bp.route("/", methods = ["GET", "PSOT"])
def dashboard():

    query = None
    data = mongodb.find_one(query)
    del data["_id"]

    fig = plotly_models.series_to_bar(data)
    graphJSON = json.dumps(fig, cls = plotly.utils.PlotlyJSONEncoder)

    return render_template("dashboard.html", graphJSON = graphJSON)