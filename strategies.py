import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import requests

class MomentumBenchmark:
    def __init__(self):
        tickers = list(pd.read_csv("sp500.csv")["Symbol"].unique())[:10]
        self.news_api = "ib0OSDasNTaYtmzK8qkDQgN0qx6g3cODrwYbDYcP"
        self.prices = yf.download(tickers=tickers, period="max").iloc[-2520:].dropna(axis=1)
        self.quarterly_prices = self.prices.resample("M").last()
        self.long_run_vol = self.quarterly_prices.pct_change().dropna().std()
        self.returns = self.prices.Close.pct_change().dropna()
        self.quarterly_dates = self.quarterly_prices.index[:-1]
        volume_data = self.prices.Volume
        self.tickers = {}
        self.reference_portfolio_momentum = pd.Series()
        self.reference_portfolio_low_momentum = pd.Series()
        self.momentum_portfolio_engineered = pd.DataFrame()
        self.low_momentum_portfolio_engineered = pd.DataFrame()
        self.long = {}
        self.short = {}
        self.ls = {}
        self.X = {}

        for i in self.quarterly_dates:
            print(i)
            if list(self.quarterly_dates).index(i) != 0:
                self.tickers[i] = self.get_ls_tickers(self.returns[i - datetime.timedelta(30): i], size = 50)  ##### This gives me the portfolio of long and short tickers

        for k,v in self.tickers.items():
            if k != self.quarterly_dates[-1]:
                self.get_portfolio_value(k, v, volume_data)

        X = pd.DataFrame.from_dict(self.X).transpose()
        X['state reset'] = (X['state'] == 0).cumsum()
        X['state_length'] = X.groupby('state reset')['state'].cumsum()

        sp = yf.download(tickers="^GSPC", period="max").Close.iloc[-2520:].dropna(axis=1).pct_change().dropna().resample("Q").sum().cumsum()


    def get_portfolio_value(self, k,v, volume_data):

        k_idx = list(self.quarterly_dates).index(k)
        next_idx = k_idx + 1
        next_date = self.quarterly_dates[next_idx]

        long_portfolio = self.returns[self.tickers[k][0].index].loc[k:next_date].dropna().mean(axis = 1)
        return_hhi = (((self.returns[self.tickers[k][0].index].loc[k:next_date].dropna().sum(axis = 0)/
                      self.returns[self.tickers[k][0].index].loc[k:next_date].dropna().sum(axis = 0).sum()) * 100)**2).sum()
        self.momentum_portfolio_engineered = pd.concat([self.momentum_portfolio_engineered, long_portfolio], axis = 0)

        short_portfolio = self.returns[self.tickers[k][1].index].loc[k:next_date].dropna().mean(axis = 1)
        self.low_momentum_portfolio_engineered = pd.concat([self.low_momentum_portfolio_engineered, short_portfolio], axis=0)
        news_sentiment = self.news_sentiment_analysis(k.tz_localize("UTC").isoformat(), next_date.tz_localize("UTC").isoformat())

        self.X[k] = {"vol_market":self.returns.loc[k:next_date].dropna().std().max(),
                    "vol_spread":long_portfolio.std() - short_portfolio.std(),
                     "vol_longrun_ratio": long_portfolio.std() / self.long_run_vol,
                    #"liquidity_spread":self.prices[(self.tickers[k[0]]].Volume.loc[k:next_date].dropna().mean(axis = 1) - self.prices[self.tickers[k[1]]].Volume.loc[k:next_date].dropna().mean(axis = 1),
                    "momentum_liquidity": volume_data[self.tickers[k][0].index].loc[k:next_date].dropna().sum(axis = 1).sum() //1000,
                    "state":1 if long_portfolio.sum() > short_portfolio.sum() else 0,
                     "rt_hhi": return_hhi,
                     "rt": long_portfolio.sum() - short_portfolio.sum()}


    def get_ls_tickers(self, data, size):

        rtq = data.cumsum().iloc[-1]
        long, short = rtq.nlargest(size), rtq.nsmallest(size)

        #ptf_long = np.round((value / 10) / self.quarterly_prices.loc[date, long],0)
        #ptf_short = np.round((value / 10) / self.quarterly_prices.loc[date, short], 0)

        return long, short

    def news_sentiment_analysis(self, start, end):

        url = "https://api.marketaux.com/v1/news/all"
        params = {
            "api_token": self.news_api,
            "language": "en",
            "published_after": "2026-05-25T00:00:00",#.tz_convert(None).strftime("%Y-%m-%dT%H:%M:%S"),
            "published_before": "2026-07-25T00:00:00",#.tz_convert(None).strftime("%Y-%m-%dT%H:%M:%S"),
            "limit": 3
        }

        response = requests.get(url, params=params)
        data = response.json()

        return data.get("data", [])

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
