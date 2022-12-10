import alpaca_trade_api as alpaca
import pandas as pd
import numpy as np
import yfinance as yf
from toolbox import yield_calc, get_price

from dotenv import dotenv_values
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
config = dotenv_values(BASE_DIR/ ".env")

api_key = config['ALPACA_PAPER_KEY']
api_secret = config['ALPACA_PAPER_SECRET_KEY']
endpoint = config['ALPACA_ENDPOINT_PAPER']

def get_assets(asset_class = None, symbol = None):

    """asset class argument includes stocks / crypto, etfs included in stocks"""

    api = alpaca.REST(api_key, api_secret, endpoint, api_version='v2')
    asset_list = api.list_assets(status = "active", asset_class=asset_class)

    return asset_list


def filter_assets(asset_list, filter = str):

    FilteredList = []

    for k in asset_list:
        if filter in k._raw["name"].lower():
            FilteredList.append(k)

    return FilteredList

def generate_ticket(symbol, qty, side):

    ticket_schema = {
        "symbol" : symbol,
        "qty": qty,
        "side" : side,
        "type": "market"
    }

    return ticket_schema

def order(ticket):

    api = alpaca.REST(api_key, api_secret, endpoint, api_version='v2')
    api.submit_order(
        symbol = ticket["symbol"],
        qty=ticket["qty"],
        side=ticket["side"],
        type=ticket["type"],
    )

def positions():

    endpoint = 'https://paper-api.alpaca.markets'
    api = alpaca.REST(api_key, api_secret, endpoint, api_version='v2')
    portfolio = api.list_positions()

    return portfolio

def get_data(symbol):

    api = alpaca.REST(api_key, api_secret, endpoint, api_version='v2')
    data = api.get_bars(symbol, "1Day", start = "2000-01-01", adjustment="all").df.close

    return data

def LongDiv_trigger():
    pass

def ShortDiv_trigger():
    pass


if __name__ == '__main__':

    #user = user()
    #user.calibrate()
    #user.backtest()
    get_assets(asset_class=None)





