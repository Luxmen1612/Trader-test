import pandas as pd
import yfinance as yf
import pandas
import numpy as np
import selenium
import requests
from bs4 import BeautifulSoup
import datetime as dt
import re

def get_year(series): #transform series index to year value

    lst = []
    idx = pd.Series(series.index.values)
    for i in idx:
        d = i.year
        lst.append(d)

    series.index = pd.Series(lst)
    return series

def yield_calc(symbol, frequency = "Q"):

    ticker = yf.Ticker(symbol)
    dividends = ticker.dividends.resample(frequency).sum()
    price = yf.download(symbol).Close.resample(frequency).last()[dividends.index.values[0]:]
    #dividends = get_year(dividends)

    div_yield = dividends / price
    index = div_yield.index.values

    div_yield = pd.Series(np.where(div_yield>0.1,0, div_yield), index=index)

    return div_yield

def rolling_vol(symbol, window):

    vol = yf.download(symbol).Close.pct_change().rolling(window = window).std()

    return vol

def get_price(symbol, year):

    price = yf.download(symbol).Close.resample('Y').last()[year]

    return price

def datetime_to_numpy(date):

    return np.datetime64(pd.Timestamp(date))

def tail_analytics(data):

    norm_excess = 0
    _ret = data.pct_change().dropna()
    _M0 = len(_ret)
    _M1 = np.average(_ret)
    _vol = np.std(_ret)
    _M3 = np.sum((_ret - _M1) ** 3) / _M0
    _M4 = np.sum((_ret - _M1) ** 4) / _M0
    kurtosis = _M4 / _vol**4 -3
    skewness = _M3 / _vol**3

    priips_var = (_vol * np.sqrt(20)) * \
                      ((-1.96 + 0.474 * skewness / np.sqrt(20)) \
                       - 0.0687 * kurtosis / 20 \
                       + 0.146 * (skewness ** 2 / 20)) \
                       - 0.5 * _vol ** 2 * 20

    return kurtosis - norm_excess, priips_var

def create_market_portfolio(data):

    nav_data = data.apply(sum, axis = 1)

    return nav_data

def retention_rate(data, date, previous_date):

    old_ptf = data.get(previous_date).get("long")
    new_ptf = data.get(date).get("long")

    retention = len(list(set(old_ptf) & set(new_ptf))) / len(old_ptf)

    return retention

def _Brownian(N, T=1, seed = np.random.randint(10**9)):
    np.random.seed(seed)
    dt = T / N  # time step
    b = np.random.normal(0., 1., int(N)) * np.sqrt(dt)  # brownian increments
    W = np.cumsum(b)  # brownian path
    return W, b


def calc_dict_spread(dict):
    long = pd.Series(dict[0]) - 1
    short = pd.Series(dict[1]) - 1
    spread = long - short

    return spread

if __name__ == '__main__':

    #symbols = ['BATS.L', '^GSPC', 'BATS.L', '^GSPC', 'BATS.L', '^GSPC', 'BATS.L', '^GSPC', 'BATS.L', '^GSPC', 'BATS.L', '^GSPC',]
    #b = get_momentum(symbols)
    symbol = "AAPL"



