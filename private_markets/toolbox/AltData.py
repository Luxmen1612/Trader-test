import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import pickle

class AltDataAnalytics:
    def __init__(self, uri = "altdata.xlsx"):

        with open("altdata.pickle", 'rb') as handle:
            self.data = pickle.load(handle)

        self.data = self.data[(self.data["Vintage"] >= 2010) & (self.data["Fund Status"] == "Liquidated")]
        self.funds = list(set(self.data["Fund ID"]))
        self.commitment = 10000000
        self.refData = yf.download("^GSPC").Close.resample("Q").last()
        self.autocorr_dict = {}
        self.multiple_dict = {}

        for f in self.funds:
            drawdowns = self.data[(self.data["Fund ID"] == f) & (self.data["Transaction Category"] == "Capital Call")]["Transaction Amount"]
            distributions = self.data[(self.data["Fund ID"] == f) & (self.data["Transaction Category"] == "Distribution")]["Transaction Amount"]
            #dates = self.data[(self.data["Fund ID"] == f) & (self.data["Transaction Category"] == "Capital Call")]["Transaction Date"]
            uncalled_capital = (self.commitment + np.cumsum(drawdowns)).shift(1).fillna(self.commitment)
            propDrawdown = drawdowns / uncalled_capital
            #bm = self.refData.loc[dates.values[0]:]
            self.multiple_dict[f] = self.calc_multiple(drawdowns, distributions)
            self.autocorr_dict[f] = self.DrawDown_GARCH(propDrawdown)

        df = pd.concat([pd.Series(self.multiple_dict), pd.Series(self.autocorr_dict)], axis=1).dropna()
        corr = np.corrcoef(df[0], df[1])

    def calc_multiple(self, drawdowns, distributions):

        try:
            multiple = np.sum(distributions) / np.sum(np.abs(drawdowns))
        except:
            multiple = 0

        return multiple

    def DrawDown_GARCH(self, drawdowns):

        autocorr = None
        try:
            autocorr = np.corrcoef(drawdowns[1:], drawdowns.shift(1).dropna()).min()

        except:
            pass

        return autocorr

if __name__ == "__main__":

    AltDataAnalytics()