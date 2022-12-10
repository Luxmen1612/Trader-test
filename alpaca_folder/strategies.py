import pandas as pd
import yfinance
import numpy as np
import matplotlib.pyplot as plt

from alpaca_folder.alpaca import get_assets, get_data
from toolbox import yield_calc
from pathlib import Path


lookback = 20
asset_list = get_assets(asset_class="us_equity")

BASE_DIR = Path(__file__).resolve().parent.parent
symbol_list = pd.read_excel(BASE_DIR/"test.xlsx", engine = "openpyxl")[0]
nsdq_tickers = pd.read_excel(BASE_DIR/"NSDQ.xlsx", engine = "openpyxl", header = None)[0]
nsdq_10_tickers = ["AAPL", "MSFT", "AMZN", "GOOG", "NVDA", "TSLA", "PEP", "COST", "META"]


def get_momentum(symbol, frequency = "Q"):

 data = get_data(symbol).resample(frequency).last()
 ret = (data / data.shift(-1)).dropna()

 return ret

def div_calibrate():

    global_df = pd.DataFrame()

    for k in symbol_list[:10]:
        try:
            div_yield = yield_calc(k, frequency = "Q")
            global_df[k] = div_yield
        except:
            pass

    return global_df.fillna(0)

def momentum_calibrate(): #works but performance speed issues might want to try concat

    global_df = pd.DataFrame()
    for k in symbol_list[:10]:
        try:

            momentum = get_momentum(k, frequency = "Q")
            global_df[k] = momentum
        except:
            pass
    return global_df.fillna(0)

def create_allocation(div_df, mom_df, n):

    portfolio = {}
    allocation = 10000

    df_slice_div = div_df.loc[n]
    df_slice_mom = mom_df.loc[n]

    ref_div_yield = np.percentile(df_slice_div, 90)
    ref_mom = np.percentile(df_slice_mom, 90)

    long_div = df_slice_div[df_slice_div > ref_div_yield].dropna().index.values
    mom_df = df_slice_mom[df_slice_mom > ref_mom].dropna().index.values

    for symbol in long_div:
        weight = allocation / len(long_div)
        price = get_data(symbol).resample("Q").last()[n]
        position = weight / price
        portfolio[symbol] = position

    for symbol in mom_df:
        weight = allocation / len(mom_df)
        price = get_data(symbol).resample("Q").last()[n]
        position = weight / price
        portfolio[symbol] = - position

    return portfolio

class portfolio_backtesting:

    def __init__(self):

        self.df = {}
        self.base_df = momentum_calibrate()
        self.div_df = div_calibrate()

    def run(self):

        for k in self.base_df.index.values:
            result = create_allocation(self.div_df, self.base_df, k)
            self.df[k] = result

    def test_perf(self):

        leverage = 1
        portfolio_nav = 0

        for k in self.df.keys():
            for item in self.df[k]:
                position = get_data(item).resample("Q").last()[k] * leverage
                portfolio_nav += position


#### NEW STRATEGY ###
""" Inverse NASDAQ components"""
def nasdaq_top_10_ew_contribution():

    nasdaq_ret = []
    portfolio_ret = []
    df = pd.DataFrame()
    interval = 8
    corr = []

    nasdaq = yfinance.download("^IXIC").Close.resample("Q").last().pct_change().dropna()
    nasdaq.index = nasdaq.index.tz_localize(None)

    df = pd.concat([df, nasdaq], axis = 1)

    for k in nsdq_10_tickers:
        data = get_data(k).resample("Q").last().pct_change().dropna()
        data.name = k
        data.index = data.index.tz_localize(None)
        df = pd.concat([df, data], axis = 1)

    df = df.fillna(0).sort_index()

    for k in df.index.values:
        nasdaq_ret.append(df.loc[k].iloc[0])
        portfolio_ret.append(np.average(df.loc[k].iloc[1:]))

    for x in range(len(nasdaq)):
        corr.append(np.corrcoef(nasdaq_ret[x:x+interval], portfolio_ret[x:x+interval]).min())

    pd.Series(corr).dropna().plot()
    plt.show()

def trade_top_10_nasdaq(leverage = 1):

    df = pd.DataFrame()
    for k in nsdq_10_tickers:
        data = get_data(k).resample("Q").last()
        data.name = k
        df = pd.concat([df, data], axis=1)

    df = df.fillna(0)

    portfolio_value = {}

    for k in df.index.values:
        position = - (df.loc[k]) * leverage
        portfolio_value[k] = np.sum(position)

    series = pd.Series(portfolio_value)

if __name__ == '__main__':

    #df = trade_top_10_nasdaq(leverage = 1)
    x = portfolio_backtesting()
    x.run()
    x.test_perf()
