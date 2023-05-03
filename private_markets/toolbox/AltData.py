import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import pickle
import datetime as dt

class AltDataAnalytics:
    def __init__(self, uri = "altdata.xlsx"):

        with open("altdata.pickle", 'rb') as handle:
            self.data = pickle.load(handle)
            self.data = self.data.set_index(self.data["Transaction Date"])

        self.data = self.data[(self.data["Vintage"] >= 2010) & (self.data["Fund Status"] == "Liquidated")]
        self.funds = list(set(self.data["Fund ID"]))
        self.commitment = 10000000
        self.refData = yf.download("^GSPC").Close.resample("Q").last()
        self.autocorr_dict = {}
        self.multiple_dict = {}
        self.draw_rate_mean = {}
        for f in self.funds:
            drawdowns = self.data[(self.data["Fund ID"] == f) & (self.data["Transaction Category"] == "Capital Call")]["Transaction Amount"]
            distributions = self.data[(self.data["Fund ID"] == f) & (self.data["Transaction Category"] == "Distribution")]["Transaction Amount"]
            uncalled_capital = (self.commitment + np.cumsum(drawdowns)).shift(1).fillna(self.commitment)
            propDrawdown = drawdowns / uncalled_capital
            self.draw_rate_mean[f] = propDrawdown.fillna(0).mean()
            self.multiple_dict[f] = self.calc_multiple(drawdowns, distributions)
            self.autocorr_dict[f] = self.DrawDown_analytics(propDrawdown)

        df = pd.DataFrame(self.multiple_dict)

    def calc_multiple(self, drawdowns, distributions):

        analytics_dict = {}
        try:
            multiple = np.sum(distributions) / np.sum(np.abs(drawdowns))
            analytics_dict["multiple"] = multiple
            analytics_dict["duration"] = (distributions.index[-1] - distributions.index[0]).days
            analytics_dict["left-right-concentration"] = np.sum(distributions[:int(np.floor(len(distributions)/2))]) / np.sum(distributions[int(np.floor(len(distributions)/2)):])


        except:
            pass

        return analytics_dict

    def DrawDown_analytics(self, drawdowns):

        autocorr = None
        try:
            autocorr = np.corrcoef(drawdowns[1:], drawdowns.shift(1).dropna()).min()

        except:
            pass

        return autocorr


if __name__ == "__main__":

    AltDataAnalytics()