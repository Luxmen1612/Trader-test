# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 16:04:50 2019

@author: chris
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#from frqresources.data import datareader
from scipy.optimize import minimize

from toolbox import _Brownian
from private_markets import jcurve as buchner_main
#from frqpriips.analytics.priips.models.pe.cat1 import strategy_map


class Brownian:  # (buchner 3.3)

    def __init__(self, context, draw_rate=0.4, draw_vol=0.02, draw_corr=0.5, seed=1):
        self.draw_corr = draw_corr
        self.draw_vol = draw_vol
        self.draw_rate = draw_rate
        self.seed = seed
        self.context = context
        _, self.e1 = _Brownian(self.context.npoints, self.context.lifespan, self.seed)

    def calc_draw_rate(self, k):
        '''
        calculate draw rate for given point in time
        '''
        average_hp = 12 * 4
        observed_hp = self.context.npoints
        draw_rate = max(1, self.draw_rate * average_hp / observed_hp)

        self.dd_noise = self.draw_corr * self.context.market_noise[k] + \
                        np.sqrt((1 - self.draw_corr ** 2)) * self.e1[k]  # 3.6

        self.draw_dynamics = max(0, (draw_rate + (self.draw_vol * self.dd_noise  # 3.7
                                                * np.sqrt(k * self.context.dt))))

        return self.draw_dynamics


    def update_dD(self, obj):

        obj.draw_dynamics = self.calc_draw_rate(obj.k)
        obj.dD = obj.draw_dynamics * (obj.comm_capital - obj.Dk) * obj.context.dt
        return obj.dD

    def residuals_for_optimization(self, params):

        draw_rate, draw_vol, draw_corr = params

        dt = 0.25
        D = []
        Dk = 0.
        draw_cls = Brownian(self.context, draw_rate, draw_vol, draw_corr, seed=1)
        for i in range(self.context.npoints):
            dr = draw_cls.calc_draw_rate(i)
            dD = dr * (1 - Dk) * dt
            D.append(dD)
            Dk += dD
        return ((pd.Series(D) - self.benchmark).dropna() ** 2).sum()
        # return dD

    def get_optimised_parameters(self):
        """
        loops through all sub Strategies, collects the data and finds the
        optimised draw_rate,draw_volatility,draw_correlation for the given draws of that strategy

        :return: list(draw_rate,draw_volatility,draw_correlation)
        """
        initial_guess = [0.41, 0.1, 0.5]
        bnds = ((0, .3), (0.0, 0.5), (0, 1.))
        _df = pd.DataFrame()
        for sub_strat in strategy_map[self.context.strategy]:
            substrat_df = datareader.Preqin.get_funds_filtered(category=sub_strat.lower(), vintage=('>', 2005), status='closed',
                                    size=('>', 200), region='', output='contribution_cum')
            substrat_df = (substrat_df[substrat_df.ffill() < 0].dropna(axis=1, how='all') / -10000000).mean(axis=1).fillna(0).ffill()
            _df = pd.concat([_df,substrat_df],axis=1)

        self.benchmark = _df.mean(axis=1)
        res = minimize(self.residuals_for_optimization, initial_guess, bounds=bnds)
        print('calculated optimised draw params for %s' %self.context.strategy)
        return res.x