import requests
import json
import pandas as pd
from dotenv import dotenv_values
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
config = dotenv_values(BASE_DIR/ ".env")

paper_trade_key = config['PAPER_TRADE_KEY']
market_data_key = config['MARKET_DATA_KEY']

#for all
order_dict = {
    "isin": "",
    "expires_at": "1D",
    "side": "",
    "quantity": 1,
    "venue": "XMUN"
}

def bulk_order():

    df = pd.read_excel("test.xlsx", engine="openpyxl")

    for i in df.index.values:

        side = df[i]['side']
        isin = df[i]
        trade(isin, df[i]["quantity"], side)


def retrieve_orders():

    endpoint = "https://paper-trading.lemon.markets/v1/orders"
    headers = {"Authorization": f"Bearer {paper_trade_key}"}

    order_log = requests.get(endpoint, headers = headers)

    return order_log.json()


def trade(isin, quantity, side = "buy"):

    order_endpoint = "https://paper-trading.lemon.markets/v1/orders"
    headers = {"Authorization": f"Bearer {paper_trade_key}"}

    order_dict['isin'] = isin
    order_dict['side'] = side
    order_dict['quantity'] = quantity

    trade = requests.post(order_endpoint, data = json.dumps(order_dict), headers = {"Authorization": f"Bearer {paper_trade_key}"}).json()
    activate_endpoint = f"https://paper-trading.lemon.markets/v1/orders/{trade.get('results').get('id')}/activate"
    activate_trade = requests.post(activate_endpoint, data = json.dumps({"pin": "7652"}), headers=headers)

    return activate_trade.json()


def all_instruments_loop(flag_var):

    response = requests.get("https://data.lemon.markets/v1/instruments/",
                            headers={"Authorization": f"{market_data_key}"})
    num_pages = response.json()['pages']
    for k in range(num_pages):
        print(k)
        response = requests.get(f"https://data.lemon.markets/v1/instruments/?page={k+1}",
                                headers={"Authorization": f"{market_data_key}"})
        data = response.json()
        for v in data['results']:
            print((v['name'], v["type"]))


def data(_data):

    data = {}
    data[_data["t"]] = _data["c"]

    return data

def get_isins():

    response = requests.get("https://data.lemon.markets/v1/instruments/",
                            headers={"Authorization": f"{market_data_key}"})

    _rawdata = response.json()
    pages = _rawdata['pages']

    for i in range(1, pages):
        try:

            response = requests.get(f"https://data.lemon.markets/v1/instruments/?page={i}",
                                        headers={"Authorization": f"{market_data_key}"})
            _data = response.json()['results']
            for d in _data:
                isin.append(d['isin'])
        except:
            pass

    return isin

def get_price_df(isin_list):

    df_agg = {}
    df = {}

    for i in isin_list:
        try:
            response = requests.get(f"https://data.lemon.markets/v1/ohlc/d1?isin={i}&from=2018-11-01",
                                headers={"Authorization": f"{market_data_key}"}).json()["results"]
            for x in response:
                stamp = x["t"]
                price = x["c"]
                df[stamp] = price
            df = {}
        except:
            pass

    return df

if __name__ == '__main__':

    isin_fund = "IE00B4QNHZ41"
    isin_share = "DE0005140008"
    #order = retrieve_orders()
    buy = trade(isin = isin_share, quantity=200)
