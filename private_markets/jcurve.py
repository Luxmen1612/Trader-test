# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 16:08:03 2019

@author: chriss
"""
import datetime as dt
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
#from frqpriips.analytics.priips.helpers import modelutils
from private_markets import invest_models, draw_models, dist_models, private_debt_model, context, simulation
from private_markets.toolbox.helpers import allocate, built_realized_df
import yfinance as yf
import matplotlib.pyplot as plt
import pymongo
import matplotlib.image as image
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)
from matplotlib.animation import FuncAnimation
from matplotlib.axis import Axis

# self.img_uri = "/home/david/Downloads/Logo PRIIPs Dark transparent.png"
# logo = image.imread(self.img_uri)
# fig, ax = plt.subplots()
# ax.set_facecolor('#1d2426')
# ax.set_title("J-curve simulation")
# ax.figure.figimage(logo, 100, 350, alpha = 1, zorder = 1)
# plt.pause(15)


bm_data = yf.download("^GSPC").Close
pymongo_uri = "mongodb+srv://draths:Bremen92@cluster0.95mle.mongodb.net/?retryWrites=true&w=majority"
db = pymongo.MongoClient(pymongo_uri)["aifm_fund_rm"]
transactions_coll = db["transactions"]
fund_coll = db["funds"]

class Jcurve:
    def __init__(self, bm_data, fund_id):

        fund_data = fund_coll.find_one({"fund_id": fund_id})
        fund_transactions = allocate(transactions_coll.find({"fund_id" : fund_id})).resample("M").sum()
        fund_valuations = pd.Series(fund_data["valuations"])
        fund_valuations.index = pd.to_datetime(fund_valuations.index, format = "%Y-%m-%d")
        fund_valuations = fund_valuations.resample("M").last().ffill()
        self.start_date = fund_data["launch date"] if len(fund_transactions) == 0 else fund_transactions.index[-1]
        self.end_date = fund_data["launch date"]+dt.timedelta(fund_data["duration"]*365)
        self.bm_data = bm_data
        self.context = context.Context(bm_data, strategy = fund_data["strategy"], rhp = 10, freq = "M", rf_rate = 0.01, comm_capital = fund_data["commitment"], start = self.start_date, end = self.end_date)

        target = fund_data["target"]
        self.beta = invest_models.PME_Buchner(self.context, self.bm_data, seed = 1).get_optimised_parameters(target)[0]
        self.realized_calls = np.abs(fund_transactions[fund_transactions < 0].sum())

        self.dict = {}
        self.navs = []
        self.P = {}

        self.realized_df, self.combined_df = built_realized_df(self.context, fund_valuations, fund_transactions)


        for i in range(2):
            sim_dict = self.simulate(10, i)
            self.navs.append(sim_dict["P"].iloc[-1])
            self.dict[i] = sim_dict
            self.P[i] = sim_dict["P"]

        self.df = pd.concat([self.combined_df, pd.DataFrame(self.P)])


    def simulate(self, degrees_freedom, item):

            scenarios_dict = {}
            number_scenarios = 1
            for seed in range(0, number_scenarios):
                #seed = item
                #print(f'running scenario {seed} for process {degrees_freedom}')
                #scenario_obj = simulation.set_params(bm_data = self.benchmark_data, seed = seed, beta = self.beta, rhp = self.rhp, env = self.env, capital=self.capital, start = )
                #scenario_obj = simulation.set_params(self.context, env = "normal", realized_calls=self.realized_calls)
                scenario_obj = simulation.set_params(self.context, env="normal", realized_calls=self.realized_df)
                scenario_obj.simulate_path()
                scenarios_dict = scenario_obj.results_df

            return scenarios_dict


if __name__ == "__main__":

    test = Jcurve(bm_data, 1)
    plt.plot(test.realized_df["P"])
    #plt.plot(pd.DataFrame(test.P), linestyle = "dashed")
    plt.plot(test.df, linestyle = "dashed")
    plt.show()
