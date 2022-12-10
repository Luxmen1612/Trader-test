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

#def get_momentum(symbol, frequency = "Q"):

#    data = get_data(symbol).resample(frequency).last
#    ret = (data / data.shift(-1)).dropna()

#    return ret

def get_price(symbol, year):

    price = yf.download(symbol).Close.resample('Y').last()[year]

    return price


if __name__ == '__main__':

    #symbols = ['BATS.L', '^GSPC', 'BATS.L', '^GSPC', 'BATS.L', '^GSPC', 'BATS.L', '^GSPC', 'BATS.L', '^GSPC', 'BATS.L', '^GSPC',]
    #b = get_momentum(symbols)
    symbol = "AAPL"
    get_momentum(symbol)
    #yield_calc(symbol)


