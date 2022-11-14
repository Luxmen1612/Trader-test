import yfinance as yf
import pandas as pd
from dotenv import dotenv_values
from pathlib import Path
import requests, json

BASE_DIR = Path(__file__).resolve().parent
config = dotenv_values(BASE_DIR/ ".env")

market_data_key = config['MARKET_DATA_KEY']
headers = {"Authorization": f"Bearer {market_data_key}"}

def get_stocks():

    endpoint = "https://data.lemon.markets/v1/instruments?type=stock"

    data = requests.get(endpoint, headers=headers).json()

    return data