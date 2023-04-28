import pandas as pd
import numpy as np
import pymongo
import matplotlib.pyplot as plt
from private_markets.jcurve import Calc
from dateutil.relativedelta import relativedelta
import yfinance as yf

pymongo_uri = "mongodb+srv://draths:Bremen92@cluster0.95mle.mongodb.net/?retryWrites=true&w=majority"
db = pymongo.MongoClient(pymongo_uri)["aifm_fund_rm"]
transactions_coll = db["transactions"]
fund_coll = db["funds"]

class fund_rm:
    def __init__(self, fund_name, benchmark):

        transactions = {}
        fund_data = fund_coll.find_one({"fund name":fund_name})
        committed_capital = fund_data["committed Capital"]
        transaction_data = transactions_coll.find({"fund":fund_name})

        for k in transaction_data:
            transactions[k["Date"]] = -k["capital call volume"]

        transactions = pd.Series(transactions).resample("m").sum()
        uncalled_capital = committed_capital + np.sum(transactions)

        fc = Calc(benchmark, target = 2.4, strategy = "private_debt",
                  rhp = fund_data["Maturity"] - len(transactions)/12,
                  env = "normal",
                  start = transactions.index[-1] + relativedelta(months = 1),
                  capital= uncalled_capital)

        self.rm_plot_data = (transactions, fc)

    def plot(self):

        realized_data = self.rm_plot_data[0]
        fc_data = self.rm_plot_data[1]

        ref = fc_data.dict[fc_data.navs.index(np.percentile(fc_data.navs, 50))]["V"]
        plt.plot(realized_data)
        plt.plot(ref, linestyle = "--")
        plt.show()

if __name__ == "__main__":

    x = fund_rm("Test RM1", yf.download("^GSPC").Close.resample("M").last())
    x.plot()
