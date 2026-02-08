import yfinance as yf
import pandas as pd
import numpy as np
import datetime


class MomentumBenchmark:
    def __init__(self):
        tickers = list(pd.read_csv("sp500.csv")["Symbol"].unique())
        self.prices = yf.download(tickers=tickers, period="max").iloc[-2520:].dropna(axis=1)
        self.quarterly_prices = self.prices.resample("M").last()
        self.returns = self.prices.Close.pct_change().dropna()
        self.quarterly_dates = self.quarterly_prices.index
        self.tickers = {}
        self.reference_portfolio_momentum = pd.Series()
        self.reference_portfolio_low_momentum = pd.Series()
        self.momentum_portfolio_engineered = pd.DataFrame()
        self.low_momentum_portfolio_engineered = pd.DataFrame()
        self.long = {}
        self.short = {}
        self.ls = {}

        for i in self.quarterly_dates:
            print(i)
            if list(self.quarterly_dates).index(i) != 0:
                self.tickers[i] = self.get_ls_tickers(self.returns[i - datetime.timedelta(20): i], size = 50)  ##### This gives me the portfolio of long and short tickers

        for k,v in self.tickers.items():
            X = self.get_portfolio_value(k,v)


        X = pd.DataFrame.from_dict(X)
        X['state reset'] = (X['state'] == 0).cumsum()
        X['state_length'] = df.groupby('reset')['state'].cumsum()

        #spread = (self.momentum_portfolio_engineered - self.low_momentum_portfolio_engineered).resample("M").sum()
        #monthly_liquidity = yf.download(tickers=tickers, period="max").Volume.iloc[-2520:].dropna(axis=1).resample("M").sum()

        sp = yf.download(tickers="^GSPC", period="max").Close.iloc[-2520:].dropna(axis=1).pct_change().dropna().resample("Q").sum().cumsum()


    def get_portfolio_value(self, k,v):

        X = {} ##explanatory variables
        if list(self.quarterly_dates).index(k) < len(self.quarterly_dates)-1:
            k_idx = list(self.quarterly_dates).index(k)
            next_idx = k_idx + 1
            next_date = self.quarterly_dates[next_idx]

            long_portfolio = self.returns[self.tickers[k][0].index].loc[k:next_date].dropna().mean(axis = 1)
            self.momentum_portfolio_engineered = pd.concat([self.momentum_portfolio_engineered, long_portfolio], axis = 0)

            short_portfolio = self.returns[self.tickers[k][1].index].loc[k:next_date].dropna().mean(axis = 1)
            self.low_momentum_portfolio_engineered = pd.concat([self.low_momentum_portfolio_engineered, short_portfolio], axis=0)

            X[k] = {"vol_market":self.returns.loc[k:next_date].dropna().std().max(),
                    "vol_spread":long_portfolio.std() - short_portfolio.std(),
                    "liquidity_spread":self.data[self.tickers[k[0]]].Volume.loc[k:next_date].dropna().mean(axis = 1) - self.data[self.tickers[k[1]]].Volume.loc[k:next_date].dropna().mean(axis = 1),
                    "state":1 if long_portfolio.sum() > short_portfolio.sum() else 0}

        return X


    def get_ls_tickers(self, data, size):

        rtq = data.cumsum().iloc[-1]
        long, short = rtq.nlargest(size), rtq.nsmallest(size)

        #ptf_long = np.round((value / 10) / self.quarterly_prices.loc[date, long],0)
        #ptf_short = np.round((value / 10) / self.quarterly_prices.loc[date, short], 0)

        return long, short


def build_index(portfolio, quarterly_prices, portfolio_value):

    dates = list(portfolio.keys())
    for k,v in portfolio.items():
        long_positions = np.round(portfolio_value / quarterly_prices.loc(k, v["long"]),0)
        short_positions = np.round(portfolio_value / quarterly_prices.loc(k, v["short"]),0)


##### NEW STATE FORECASTING MODEL ####
momentum_ticker = "SPMO"
value_ticker = "SPVV"




if __name__ == "__main__":

    x = MomentumBenchmark()
