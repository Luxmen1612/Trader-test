from flask import Flask

from api.app import init_app

app = init_app()

if __name__ == "__main__":
    app.run()