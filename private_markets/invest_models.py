# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 16:22:23 2019

@author: chris
"""
import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import datetime as dt

#from frqresources.data import datareader
#from frqpriips.analytics.priips.models.pe.cat1 import strategy_map
from toolbox import _Brownian
from private_markets.toolbox.helpers import cat3_rndm_selector
from private_markets import simulation

import yfinance as yf

CONFIG_PATH = os.path.abspath(__file__)
ROOT = os.path.dirname(os.path.dirname(CONFIG_PATH))

#
class PME_Buchner:
    '''
    benchmark index adjusted for alpha/beta + idio vol
    '''

    def __init__(self, context, bm, alpha_mu=0., beta=1.,
                 alpha_vol=0., seed=None, df=3, perf_as_pct_comm_cap = 0.4):

        self.perf_as_pct_comm_cap = perf_as_pct_comm_cap
        self.df = df
        self.context = context
        self.bm = bm
        #self.benchmark_prices = datareader.Linear.get_ohlc(self.context.benchmark_ric).close
        self.benchmark_prices = self.bm
        self.alpha_mu = alpha_mu
        self.beta = beta
        self.alpha_vol = alpha_vol
        self.seed = seed
        self.ret_k = 0.
        _, self.e = _Brownian(self.context.npoints, self.context.lifespan, self.seed)

        self._calculate_PME_returns()


    def _calculate_PME_returns(self):

        self.benchmark_prices = self.benchmark_prices.drop_duplicates()[-1600:]
        self.returns = self.benchmark_prices.resample("M").last().pct_change().fillna(0)
        self.sd = self.benchmark_prices.resample("M").last().pct_change().fillna(0).std()

        # reindex the benchmark to frequency and timeframe defined in context
        #self.benchmark_prices_reindexed = self.benchmark_prices.reindex(self.context.stamp, method='pad')
        #self.returns = self.benchmark_prices_reindexed.pct_change().fillna(0)  # check filling


    def update_Vk(self,obj): #original

        alpha_ret = (self.alpha_mu * obj.context.dt + self.alpha_vol * self.e[obj.k] * np.sqrt(self.context.dt))
        #beta_ret = self.beta * self.returns #(self.returns + self.sd * np.random.standard_t(10))   #self.returns[obj.stamp]
        beta_ret = self.beta * cat3_rndm_selector(self.returns)[0]

        print(alpha_ret + beta_ret)

        obj.Vk = obj.Vk *(1 + (alpha_ret + beta_ret)) - obj.dR + obj.dD

        #obj.Vk = obj.Vk * (1+
        #                   self.alpha_mu * obj.context.dt +
        #                    self.beta[0] * self.returns[obj.stamp] +

        #              self.alpha_vol *  self.e[obj.k] * np.sqrt(self.context.dt))    \
        #              - obj.dR + obj.dD


    def residuals_for_optimization(self, params, target):

        """"returns the sum of differences ^2 between the cumulated invest model return timeseries
            and the SPX returns given an alpha_mu and alpha_vol"""
        from private_markets import draw_models, dist_models
        #alpha_mu, alpha_vol, beta = params
        alpha_mu = 0
        alpha_vol = 0
        beta = params

        dist_obj = dist_models.Brownian(self.context, 0.02, 0.03, 0.8, seed=1)
        draw_obj = draw_models.Brownian(self.context,0.1, 0.2, 0.5, seed=1)
        invest_obj = PME_Buchner(self.context, self.bm, alpha_vol=alpha_vol,
                         alpha_mu=alpha_mu, beta=beta, seed=3, df=10)
        simulation_obj = simulation.Simulation(self.context, draw_obj, dist_obj, invest_obj, realized_calls=0, env = "beta_sourcing")
        simulation_obj.simulate_path()
        #
        #target_value = 2.
        target_value = target
        #return ((pd.Series(((simulation_obj.Pk_lst / simulation_obj.Pk_lst[0]) - 1))
        #         - self.returns.cumsum()).dropna() ** 2).sum()

        return (target_value - pd.Series(simulation_obj.Pk_lst / simulation_obj.Pk_lst[0])[-1])**2



    def get_optimised_parameters(self, target):
        """
        minimizes residuals_for_optimization method for alpha_mu and alpha_vol
        :return: list(dist_rate,dist_volatility,dist_correlation)
        """
        #initial_guess = [0.0,0.0,1.]
        #bnds = ((-0.1,0.1), (0.,0.2),(0.,3.))
        initial_guess = [1.]
        bnds = ((0., 2.), )

        res = minimize(self.residuals_for_optimization, initial_guess, bounds=bnds, args = target)
        print('calculated optimised draw params for %s' %self.context.strategy)
        return res.x