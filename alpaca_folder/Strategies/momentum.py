from alpaca_folder.alpaca import get_data, generate_ticket, order
import pandas as pd
import numpy as np
from pathlib import Path
from toolbox import calc_dict_spread, retention_rate

BASE_DIR = Path(__file__).resolve().parent.parent.parent
symbol_list = pd.read_excel(BASE_DIR/"test.xlsx", engine = "openpyxl")[0]

class Momentum:
    def __init__(self):

        self.data = {}
        self.portfolios = {}
        self.percentile_lst = [50, 60, 70, 80, 90]
        self.percentile_dict = {}
        self.percentile_spread = {}
        self.retention = {}
        self.spread = {}

        for k in symbol_list:
            self.data[k] = get_data(k)

        self.df = pd.DataFrame(self.data).resample("M").last()
        for perc in self.percentile_lst:
            self.create_portfolio(percentile = perc)
            x = self.calc_return()
            self.percentile_dict[perc] = x
            self.percentile_spread[perc] = calc_dict_spread(x)

    def create_portfolio(self, percentile):

        returns = (self.df / self.df.shift(1)).dropna() -1
        dates = list(returns.index.values)
        for k in dates:
            df_slice = returns.loc[k]
            threshold_long = np.percentile(df_slice, percentile)
            threshold_short = np.percentile(df_slice, 100-percentile)
            long = df_slice[df_slice > threshold_long]
            short = df_slice[df_slice < threshold_short]
            self.portfolios[k] = {"long": long, "short": short}
            if dates.index(k) != 0:
                self.retention[k] = retention_rate(self.portfolios, k, dates[dates.index(k)-1])

    def place_order(self):
        pass

    def spread_analytics(self):

        self.spread_dict = {{}}
        for k in self.percentile_dict.keys():
            data = pd.Series(self.percentile_dict[k])
            self.spread_dict["average"] = np.average(data)
            self.spread_dict["cumsum"] = np.cumprod(1+data)


    def calc_return(self):

        long_nav = {}
        short_nav = {}
        long_ret = []
        short_ret = []
        spread = {}

        dates = list(self.portfolios.keys())
        for k in range(len(dates)):
            if k != len(dates)-1:
                for item in self.portfolios[dates[k]]["long"].index.values:
                    ret_l = self.df[item].loc[dates[k]] / self.df[item].loc[dates[k+1]]
                    long_ret.append(ret_l)

                long_nav[dates[k]] = np.average(long_ret)

                for item in self.portfolios[dates[k]]["short"].index.values:
                    ret_s = self.df[item].loc[dates[k]] / self.df[item].loc[dates[k+1]]
                    short_ret.append(ret_s)
                    spread[dates[k]] = ret_l - ret_s

                short_nav[dates[k]] = np.average(short_ret)

        return long_nav, short_nav, spread


class DivYield:
    def __init__(self):

        self.ticker = "JEPI"
        self.data()

    def data(self):

        self.data = get_data(self.ticker)
        self.returns = self.data.pct_change().dropna()
        self.norm_prices = np.cumprod(1 + self.returns)

    def place_order(self, volume = 0., direction = "buy"):

        ticket = generate_ticket(self.ticker, volume, direction)
        order(ticket)


class NSDQ_10_ptf:

    def __init__(self, freq = "M"):

        self.freq = freq
        df = {}
        tickers = ["AAPL", "MSFT", "AMZN", "GOOG", "NVDA", "TSLA", "PEP", "COST", "META"]
        for t in tickers:

            df[t] = get_data(t).resample(self.freq).last()

        ptf = pd.DataFrame(df).sum()

        return ptf


if __name__ == "__main__":

    x = Momentum()
    y = DivYield().norm_prices
    debug = 1