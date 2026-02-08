from flask import Flask
from flask import render_template

from api.app import init_app

app = init_app()

@app.route('/index')
def index():
    # show the subpath after /path/
    return render_template("index.html")

if __name__ == "__main__":
    app.run()