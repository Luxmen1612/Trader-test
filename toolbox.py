import pandas as pd
import yfinance as yf
import pandas
import numpy as np

def get_year(series): #transform series index to year value

    lst = []
    idx = pd.Series(series.index.values)
    for i in idx:
        d = i.year
        lst.append(d)

    series.index = pd.Series(lst)
    return series

def yield_calc(symbol, year):

    ticker = yf.Ticker(symbol)
    price = get_year(yf.download(symbol).Close.resample('Y').last())
    dividends = ticker.dividends.resample('Y').sum()
    dividends = get_year(dividends)

    div_yield = dividends[year] / price[year]

    if div_yield > 0.1:
        div_yield = 0.

    return div_yield

def rolling_vol(symbol, window):

    vol = yf.download(symbol).Close.pct_change().rolling(window = window).std()

    return vol

def get_momentum(symbol, year):

    prior_year = year - np.timedelta64(365, 'D')

    if str(prior_year)[9] != '1':
        prior_year = prior_year + np.timedelta64(1, 'D')
    elif str(prior_year)[8] == '0':
        prior_year = prior_year - np.timedelta64(1, 'D')

    data = yf.download(symbol).Close.resample('Y').last()
    ret = (data[year] / data[prior_year]) - 1

    return ret

def get_price(symbol, year):

    price = yf.download(symbol).Close.resample('Y').last()[year]

    return price


if __name__ == '__main__':

    symbols = ['BATS.L', '^GSPC', 'BATS.L', '^GSPC', 'BATS.L', '^GSPC', 'BATS.L', '^GSPC', 'BATS.L', '^GSPC', 'BATS.L', '^GSPC',]
    b = get_momentum(symbols)


