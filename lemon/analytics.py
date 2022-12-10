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

def get_stock_by_isin(isin):

    endpoint = f"https://data.lemon.markets/v1/instruments?isin={isin}"

    data = requests.get(endpoint, headers=headers).json()

    return data

def compare_inverse(factor, isin, reference_ticker):

    stock_data = get_stock_by_isin(isin)
    reference_data = yf.download(tickers=reference_ticker).Close.pct_change().dropna()

    #Find way to compare dates
