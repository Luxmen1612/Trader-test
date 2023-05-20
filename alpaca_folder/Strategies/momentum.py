from alpaca_folder.alpaca import get_data
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

        for k in symbol_list[:100]:
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
            threshold = np.percentile(df_slice, percentile)
            long = df_slice[df_slice > threshold]
            short = df_slice[df_slice < threshold]
            self.portfolios[k] = {"long": long, "short": short}
            if dates.index(k) != 0:
                self.retention[k] = retention_rate(self.portfolios, k, dates[dates.index(k)-1])


    def calc_return(self):

        long_nav = {}
        short_nav = {}
        long_ret = []
        short_ret = []

        dates = list(self.portfolios.keys())
        for k in range(len(dates)):
            if k != len(dates)-1:
                for item in self.portfolios[dates[k]]["long"].index.values:
                    ret = self.df[item].loc[dates[k]] / self.df[item].loc[dates[k+1]]
                    long_ret.append(ret)

                long_nav[dates[k]] = np.average(long_ret)

                for item in self.portfolios[dates[k]]["short"].index.values:
                    ret = self.df[item].loc[dates[k]] / self.df[item].loc[dates[k+1]]
                    short_ret.append(ret)

                short_nav[dates[k]] = np.average(short_ret)

        return long_nav, short_nav

if __name__ == "__main__":

    Momentum()