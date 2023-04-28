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
from private_markets import invest_models, draw_models, dist_models, private_debt_model
import yfinance as yf
from toolbox import _Brownian
import matplotlib.pyplot as plt
import matplotlib.image as image
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)
from matplotlib.animation import FuncAnimation
from matplotlib.axis import Axis
import numpy_financial as npf


bm_data = yf.download("^GSPC").Close

def adjust_CF_for_fees(df, entry = None, exit = None, ongoing = 0.012, transaction = None, carry = 0.2, hurdle = 0.08):

    carry_taken = {}
    adj_df = df
    distributions = adj_df["R"]
    CFs = adj_df["dC"]
    CFs[-1] += adj_df["V"].iloc[-1]

    ongoing_cost = 10000 * (ongoing + transaction)
    CFs_net = CFs - 10000 * (ongoing + transaction)
    ongoing_cost_acc = np.cumsum(pd.Series(data = np.zeros(len(CFs)) + ongoing_cost, index = CFs.index))

    irr = CFs_net.expanding(1).agg(npf.irr)
    carry_cfs = irr[irr >= hurdle]
    try:
        for k in distributions[carry_cfs.index:]:
            date = carry_cfs[carry_cfs == k]
            if len(carry_taken = 0):
                carry = (k - 10000) * carry
                carry_taken[date] = carry
            else:
                carry = (k - 10000) * carry - np.sum(carry_taken)
                carry_taken[date] = carry

        carry_taken = pd.Series(carry_taken).reindex(CFs, method = None)

    except:
        carry_taken = 0

    tot_cost_acc = carry_taken + ongoing_cost_acc
    adj_df["P"] = adj_df["P"] - tot_cost_acc

    return adj_df["P"]

def set_params(df = 10, seed = 0, beta = 1, rhp = 1,  env = "stress", strategy = "private_debt", capital = 10000.):

    context_obj = Context(bm_data, strategy = strategy, rhp = rhp, benchmark_ric = "^GSPC", freq = "M", rf_rate = 0. , comm_capital = capital, seed = seed)

    if strategy != "private_debt":

        if env == "normal":
            draw_obj = draw_models.Brownian(context_obj, draw_rate=0.1/4, draw_vol = 0.25/4, draw_corr = 0.5, seed = seed)
            dist_obj = dist_models.Brownian(context_obj, dist_rate=0.02/4, dist_vol=0.05/4, dist_corr = 0.8, dist_start_pct_of_committed=0.5)
            invest_obj = invest_models.PME_Buchner(context_obj, bm_data, alpha_vol=0.15, alpha_mu = 0.0, seed = seed, beta = beta, df = df)

        else:
            draw_obj = draw_models.Brownian(context_obj, draw_rate=0.05/4, draw_vol=0.35/4, draw_corr=0.5, seed=seed)
            dist_obj = dist_models.Brownian(context_obj, dist_rate=0.01/4, dist_vol=0.15/4, dist_corr=0.8,
                                            dist_start_pct_of_committed=0.5)
            invest_obj = invest_models.PME_Buchner(context_obj, bm_data, alpha_vol=0.25, alpha_mu=-0.01, seed=seed,
                                                   beta=beta, df=df)
    else:

        draw_obj = draw_models.Brownian(context_obj, draw_rate=0.1/4, draw_vol=0.25/4, draw_corr=0.5, seed=seed)
        dist_obj = dist_models.Brownian(context_obj, dist_rate=0.02/4, dist_vol=0.05/4, dist_corr=0.8,
                                        dist_start_pct_of_committed=0.5)
        invest_obj = private_debt_model.Kupiec(context_obj, rate = 0.1/4, pod=0.2/4, eod=1.0, lgd= 0.5, seed=seed)
    simulation_obj = Simulation(context_obj, draw_obj, dist_obj, invest_obj)

    return simulation_obj

class Calc:
    def __init__(self, bm_data, target, strategy, rhp, env, start, capital = 10000):

        self.capital = capital
        self.env = env
        self.rhp = rhp
        self.benchmark_data = bm_data
        self.context = Context(self.benchmark_data, strategy = strategy, rhp = 10, benchmark_ric = "^GSPC", freq = "M", rf_rate = 0. , comm_capital = self.capital, seed = None, start = start)
        self.target_multiple = target
        self.beta = invest_models.PME_Buchner(self.context, self.benchmark_data, seed = 1).get_optimised_parameters(target)

        self.dict = {}
        self.navs = []

        #self.img_uri = "/home/david/Downloads/Logo PRIIPs Dark transparent.png"
        #logo = image.imread(self.img_uri)
        #fig, ax = plt.subplots()
        #ax.set_facecolor('#1d2426')
        #ax.set_title("J-curve simulation")
        #ax.figure.figimage(logo, 100, 350, alpha = 1, zorder = 1)
        #plt.pause(15)

        for i in range(11):
            dict = self.simulate(10, i)
            #cost_adjusted_values = adjust_CF_for_fees(dict.copy(), 0, 0, 0.012,0, 0.2, 0.08)
            #plt.pause(0.15)
            #ax.plot(cost_adjusted_values)
            self.navs.append(dict["P"].iloc[-1])
            self.dict[i] = dict

        test = 1

    def simulate(self, degrees_freedom, item):

            scenarios_dict = {}
            number_scenarios = 1
            for seed in range(0, number_scenarios):
                seed = item
                print(f'running scenario {seed} for process {degrees_freedom}')
                scenario_obj = set_params(seed = seed, beta = self.beta, rhp = self.rhp, env = self.env, capital=self.capital)
                scenario_obj.simulate_path()
                scenarios_dict = scenario_obj.results_df

            return scenarios_dict


##########

class Context:

    def __init__(self, benchmark_data, strategy, rhp, benchmark_ric='.SPX', freq="M", rf_rate=0., comm_capital=10000., seed=None, start = None):
        self.rhp = rhp
        self.start = start
        self.benchmark_ric = benchmark_data
        self.start_dt = dt.date.today() if self.start is None else self.start
        self.end_dt = self.start_dt + relativedelta(months =self.rhp * 12)

        #self.lookback_start = self.start_dt - relativedelta(years=self.rhp)
        self.stamp = pd.date_range(freq=freq, start=self.start_dt
                                   , end=self.end_dt)

        self.freq = freq
        self.npoints = len(self.stamp)
        dt_map = {"Y": 1., "Q": 0.25, "M": 0.083333, "D": 0.002739}
        self.dt = dt_map[freq]
        self.lifespan = self.npoints * self.dt

        self.rf_rate = rf_rate
        self.seed = seed
        self.comm_capital = comm_capital
        self.strategy = strategy
        # brownian market
        _, self.market_noise = _Brownian(self.npoints, self.lifespan, self.seed)

class Simulation:

    def __init__(self, context, draw_model, dist_model, invest_model):
        # args
        self.k = 0
        self.context = context
        self.stamp = self.context.stamp
        self.draw_model = draw_model
        self.dist_model = dist_model
        self.invest_model = invest_model
        # self.alpha_model = alpha_model
        self.comm_capital = context.comm_capital
        self.Ck = context.comm_capital
        self.Vk = 0.
        self.Dk = 0.
        self.dR = 0.
        self.Rk = 0.
        self.dD_lst = pd.Series(dtype='float64')
        self.dR_lst = pd.Series(dtype='float64')
        self.dC_lst = pd.Series(dtype='float64')
        self.Vk_lst = pd.Series(dtype='float64')
        self.Pk_lst = pd.Series(dtype='float64')
        self.Ck_lst = pd.Series(dtype='float64')
        self.Dk_lst = pd.Series(dtype='float64')
        self.Rk_lst = pd.Series(dtype='float64')

    def main(self):
        # update dD from the dD_model
        self.draw_model.update_dD(self)

        # update list
        self.Dk += self.dD
        self.Dk_lst.loc[self.stamp] = self.Dk
        self.dD_lst.loc[self.stamp] = self.dD

        # update_dR
        self.dist_model.update_dR(self)

        # update list
        self.Rk += self.dR
        self.dR_lst.loc[self.stamp] = self.dR
        self.Rk_lst[self.stamp] = self.Rk

        # update_dC(self):
        self.dC = self.Ck * self.context.rf_rate * self.context.dt - self.dD + self.dR
        self.dC_lst.loc[self.stamp] = self.dC

        # update list
        self.Ck += self.dC
        self.Ck_lst.loc[self.stamp] = self.Ck

        # update_Vk
        self.invest_model.update_Vk(self)

        # update list
        self.Vk_lst.loc[self.stamp] = self.Vk

        # update_Pk(self):
        self.Pk = self.Ck + self.Vk

        # update list
        self.Pk_lst.loc[self.stamp] = self.Pk


    def calc_results_df(self):
        df = pd.DataFrame(index=self.context.stamp)

        df['dD'] = self.dD_lst
        df['D'] = self.Dk_lst
        df['dR'] = self.dR_lst
        df['R'] = self.Rk_lst
        df['dC'] = self.dC_lst
        df['C'] = self.Ck_lst
        df['V'] = self.Vk_lst
        df['P'] = self.Pk_lst

        self.results_df = df

    def simulate_path(self):
        for k, stamp in enumerate(self.context.stamp):
            self.k = k
            self.stamp = stamp
            self.main()

        self.calc_results_df()


if __name__ == "__main__":

    test = Calc(bm_data, target = 2.7, strategy = "private_debt", rhp = 10, env = "normal", start = None, capital=100000000)