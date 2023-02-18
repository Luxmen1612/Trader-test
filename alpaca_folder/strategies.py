import math

import pandas as pd
import yfinance
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

from alpaca_folder.alpaca import get_assets, get_data
from toolbox import yield_calc, create_market_portfolio, tail_analytics, retention_rate
import mongomodels
from pathlib import Path
from dotenv import dotenv_values
from sklearn.linear_model import LinearRegression
import pymongo

BASE_DIR = Path(__file__).resolve().parent.parent
config = dotenv_values(BASE_DIR/ ".env")

mongo_uri = config['MONGO_DB_URI']

db = "portfolio"
coll = "L/S"

lookback = 20
asset_list = get_assets(asset_class="us_equity")
symbol_list = pd.read_excel(BASE_DIR/"test.xlsx", engine = "openpyxl")[0]
nsdq_tickers = pd.read_excel(BASE_DIR/"NSDQ.xlsx", engine = "openpyxl", header = None)[0]
nsdq_10_tickers = ["AAPL", "MSFT", "AMZN", "GOOG", "NVDA", "TSLA", "PEP", "COST", "META"]

def calc_nav():
    pass

def regression(dependent, independent, lag):

    d_var = dependent.shift(lag).dropna() #(spread in % form already)
    i_var = yfinance.download(independent).Close.resample("Q").last().pct_change().dropna()

    i_ret = i_var[-len(d_var):]
    x = np.array(i_ret).reshape((-1, 1))
    y = np.array(d_var)
    model = LinearRegression()
    model.fit(x, y)

    return model

def construct_master_price_files(frequency = "Q"):

    df = pd.DataFrame(index = pd.to_datetime([]))
    df_nonsampled = pd.DataFrame(index = pd.to_datetime([]))
    for k in pd.Series(symbol_list.unique()):
        data_nonsampled = get_data(k).tz_localize(None)
        data_nonsampled.name = k
        data = data_nonsampled.resample(frequency).last()
        data.name = k
        df = pd.concat([df, data], axis = 1)
        df_nonsampled = pd.concat([df_nonsampled, data_nonsampled], axis = 1)

    return df.dropna(axis = 1), df_nonsampled.dropna(axis = 1)

def get_momentum(df, symbol, frequency = "Q"):

    #data = get_data(symbol).resample(frequency).last()
    data = df[symbol]
    ret = (data / data.shift(1)).dropna()
    ret.index = pd.to_datetime(ret.index)

    return ret

def div_calibrate(df):

    global_df = pd.DataFrame()

    #for k in symbol_list[:1000]:
    for k in df.columns:
        try:
            div_yield = yield_calc(k, frequency = "Q")
            div_yield.name = k
            global_df = pd.concat([global_df, div_yield], axis = 1)

        except:
            pass

    return global_df.fillna(0)

def momentum_calibrate(df): #works but performance speed issues might want to try concat

    global_df = pd.DataFrame(index = pd.to_datetime([]))
    #for k in symbol_list[:1000]:
    for k in df.columns:
        try:
            momentum = get_momentum(df, k, frequency = "Q")
            global_df = pd.concat([global_df, momentum], axis = 1)
        except:
            pass

    return global_df.fillna(0)

def create_allocation(master_df, div_df, mom_df, n):

    portfolio = {}
    long_portfolio = {}
    short_portfolio = {}

    allocation = 10000

    div_df.index = pd.DatetimeIndex(div_df.index.values)
    df_slice_div = div_df.loc[n]
    df_slice_mom = mom_df.loc[n]

    ref_div_yield = np.percentile(df_slice_div, 50)
    ref_mom = np.percentile(df_slice_mom, 50)

    long_div = df_slice_div[df_slice_div > ref_div_yield].dropna().index.values
    mom_df = df_slice_mom[df_slice_mom > ref_mom].dropna().index.values

    for symbol in long_div:

        try:
            weight = allocation / len(long_div)
            ##price = get_data(symbol).resample("Q").last()[n]
            price = master_df[symbol][n]
            position = weight / price
            long_portfolio[symbol] = (position, price)

        except:
            pass

    for symbol in mom_df:
        try:
            weight = allocation / len(mom_df)
            # price = get_data(symbol).resample("Q").last()[n]
            price = master_df[symbol][n]
            position = weight / price
            short_portfolio[symbol] = (- position, price)
        except:
            pass

    return long_portfolio, short_portfolio


class portfolio_backtesting:

    def __init__(self):

        self.master_long_portfolio = {}
        self.master_short_portfolio = {}

        self.master_df = construct_master_price_files(frequency = "Q")
        self.prices_df = self.master_df[0]
        self.prices_df_nonsampled = self.master_df[1]

        self.market_portfolio = create_market_portfolio(self.prices_df_nonsampled)
        self.momentum_kurtosis = {}
        self.momentum_var = {}
        self.market_kurtosis = {}
        self.market_var = {}
        self.divyield_retention = {}
        self.momentum_retention = {}

        self.base_df = momentum_calibrate(self.prices_df)
        self.div_df = div_calibrate(self.prices_df)

        self.long_return = {}
        self.short_return = {}

        self.reference_dates = self.base_df.index.values

       # self.mongo_client = mongomodels.MongoClient('Trader', 'L/S Portfolio')

    def run(self):

        for k in self.reference_dates:
            result = create_allocation(self.prices_df, self.div_df, self.base_df, k)

            long_portfolio = {
                'direction': "long",
                'data': result[0]
            }

            short_portfolio = {
                'direction': "short",
                'data': result[1]
            }

            self.master_long_portfolio[k] = long_portfolio
            self.master_short_portfolio[k] = short_portfolio

            self.calc_return(k, self.master_long_portfolio, self.master_short_portfolio)

    def calc_return(self, k, long, short):

        long_yields = []
        short_yields = []
        momentum_portfolio = pd.DataFrame()
        momentum_tail = {}

        if np.where(self.reference_dates == k)[0] == 0:
            previous_date = k
            tail_data = tail_analytics(self.market_portfolio[:k])
            self.market_kurtosis[k] = tail_data[0]
            self.market_var[k] = tail_data[1]


        else:
            previous_date = self.reference_dates[np.where(self.reference_dates == k)[0]-1][0]
            tail_data = tail_analytics(self.market_portfolio[previous_date:k])
            self.market_kurtosis[k] = tail_data[0]
            self.market_var[k] = tail_data[1]

        self.divyield_retention[k] = retention_rate(long, k, previous_date)
        for i in long.get(previous_date).get("data").keys():
            prices = self.prices_df[i]
            latest_price = prices[k]

            if np.where(self.reference_dates == k)[0] == 0:
                initial_price = latest_price
            else:
                initial_price = prices[previous_date]

            ret = latest_price / initial_price
            long_yields.append(ret)

        self.momentum_retention[k] = retention_rate(short, k, previous_date)
        for y in short.get(previous_date).get("data").keys():
            prices = self.prices_df[y]
            latest_price = prices[k]
            market_prices = self.prices_df_nonsampled[y]
            momentum_portfolio = pd.concat([momentum_portfolio, market_prices], axis = 1)

            if np.where(self.reference_dates == k)[0] == 0:
                initial_price = latest_price
            else:
                initial_price = prices[previous_date]

            ret = latest_price / initial_price
            short_yields.append(ret)

        self.long_return[k] = np.average(long_yields)
        self.short_return[k] = np.average(short_yields)
        momentum_ptf = momentum_portfolio.apply(sum, axis = 1)
        momentum_tail[k] = tail_analytics(momentum_ptf[previous_date:k])
        self.momentum_kurtosis[k] = momentum_tail[k][0]
        self.momentum_var[k] = momentum_tail[k][1]


    def get_spread(self):

        self.long_return = pd.Series(self.long_return)
        self.short_return = pd.Series(self.short_return)

        self.spread = self.long_return - self.short_return

        self.ndx = yfinance.download("^IXIC").Close.resample("Q").last().pct_change() * 100

        self.kurtosis_ratio = pd.Series(self.momentum_kurtosis) / pd.Series(self.market_kurtosis)
        self.var_ratio = pd.Series(self.momentum_var) / pd.Series(self.market_var)


    def stats(self, lags = [0,1,2,3], independent = "^IXIC"):

        output = {}
        self.i_var = yfinance.download(independent).Close.resample("Q").last().pct_change().dropna()

        for l in lags:
            output[f"{l}_lag_regression"] = regression(dependent=self.spread, independent = independent, lag = l)

        output["correl_long"] = np.corrcoef(self.i_var[- len(self.long_return):], self.long_return)
        output["correl_short"] = np.corrcoef(self.i_var[-len(self.short_return):], self.short_return)
        output["spred_prod"] = np.product(self.spread)
        output["index_prd"] = np.product(self.i_var[-len(self.spread):])

        return output

    def out_of_sample(self):

        _id = 0
        prev_date = self.prices_df.index.values[-2]
        result = create_allocation(self.prices_df, self.div_df, self.base_df, prev_date)
        _data = pymongo.MongoClient(mongo_uri)[db][coll].find().sort("id", -1).limit(1)
        for k in _data:
            try:
                _id = k["id"] if not None else 0
            except:
                _id = 0

        Portfolio = {
            'id': _id + 1,
            'shuffle_date': str(prev_date),
            'long': result[0],
            'short': result[1],
        }

        pymongo.MongoClient(mongo_uri)[db][coll].insert_one(Portfolio)

    def get_latest_performance(self, bm = "^GSPC"):

        prices = self.prices_df_nonsampled
        long_ret = {}
        short_ret = {}
        _oldnav = 0
        _newnav = 0
        _data = pymongo.MongoClient(mongo_uri)[db][coll].find().sort("id", -1).limit(1)

        for x in _data:
            for k in x["long"].keys():
                _oldnav += np.abs(x["long"][k][1] * x["long"][k][0])
                _newnav += np.abs(prices[k].iloc[-1] * x["long"][k][0])
                long_ret[k] = prices[k].iloc[-1] / x["long"][k][1] -1

        for w in _data:
            for v in w["short"].keys():
                _oldnav += w["short"][v][1] * w["short"][v][0]
                _newnav += prices[v].iloc[-1] * w["short"][v][0]
                short_ret[v] = prices[v].iloc[-1] / w["short"][v][1] -1

        return long_ret, short_ret, _oldnav, _newnav

def assess_return_out_sample():
    pass

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


def country_ref(country = "Italy"):

    diff = 0

    uri = f"/home/david/Documents/{country}_2023_ref.xlsx"

    df = pd.read_excel(uri, "Bonds", engine = "openpyxl")

    fixed_subset = df[df["Coupon Class"] == "Fixed Coupon"]

    for k in range(len(fixed_subset)):
        #length = fixed_subset["Maturity Date"].iloc[k] - fixed_subset["Issue Date"].iloc[k]
        coupon_diff = (fixed_subset["Yield"].iloc[k] - fixed_subset["Coupon"].iloc[k]) / 100
        budget_diff = coupon_diff * fixed_subset["Amount Outstanding"].iloc[k]
        diff += budget_diff

    zero_subset =  df[df["Coupon Class"] == "Discount/Zero Coupon"]
    outstanding_zero = zero_subset["Amount Outstanding"].sum()
    outstanding_nonzero = fixed_subset["Amount Outstanding"].sum()

    return {"fixed_diff": diff,
            "Non-zero outstanding": outstanding_nonzero,
            "Zero outstanding": outstanding_zero}



if __name__ == '__main__':

    #nasdaq_top_10_ew_contribution()
    #df = trade_top_10_nasdaq(leverage = 1)

    #country_ref()

    x = portfolio_backtesting()
    x.get_latest_performance()
    #x.run()
    #x.get_spread()
    #x.stats()
    #x.out_of_sample()
