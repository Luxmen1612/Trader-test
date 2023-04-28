import numpy as np
import pandas as pd
from scipy.optimize import minimize
#from frqresources.data import datareader
from toolbox import _Brownian
from private_markets import jcurve as buchner_main
#from frqpriips.analytics.priips.models.pe.cat1 import strategy_map


class Brownian:  # buchner

    def __init__(self, context, dist_rate=0.08, dist_vol=0.11, dist_corr=0.8,
                 dist_start_pct_of_committed=0., seed=None):

        self.dist_corr = dist_corr
        self.dist_vol = dist_vol
        self.dist_rate = dist_rate
        self.context = context
        self.dist_start_pct_of_committed = dist_start_pct_of_committed
        self.seed = seed
        _, self.e2 = _Brownian(self.context.npoints, self.context.lifespan, self.seed)

    def calc_dist_rate(self, k):  # 3.5

        self.dist_noise = self.dist_corr * self.context.market_noise[k] + \
                          (np.sqrt(1 - self.dist_corr ** 2) * self.e2[k])  # 3.7

        self.dist_dynamics = self.dist_rate * (k + 1) * self.context.dt + \
                             self.dist_vol * self.dist_noise * \
                             np.sqrt((k) * self.context.dt)  # 3.5

        return self.dist_dynamics

    def update_dR(self, obj):  # 3.1

        if obj.Dk / obj.comm_capital > self.dist_start_pct_of_committed:
            self.dist_dynamics = np.abs(self.calc_dist_rate(obj.k))
            obj.dR = self.dist_dynamics * obj.Vk * obj.context.dt
        else:
            obj.dR = 0.
        return obj.dR



    def residuals_for_optimization(self, params):
        import draw_models
        dist_rate, dist_vol, dist_corr = params
        dt = 0.25
        Dk = 0.
        R = []
        V = []
        Rk = 0.
        dR=0.
        Vk = 0.
        Vhat = 0.
        dist_cls = Brownian(self.context, dist_rate, dist_vol, dist_corr, seed=1)
        draw_cls = draw_models.Brownian(self.context,0.41, 0.21, 0.5, seed=1)

        for i in range(self.context.npoints):

            #draw model
            dk = draw_cls.calc_draw_rate(i)
            dD = dk * (1 - Dk) * dt
            Dk += dD

            vk = dist_cls.calc_dist_rate(i)
            dR = vk * Vhat * dt
            Vhat += (dD-dR)*(1.1**(i*dt))         #assumed 10IRR for compounding
            V.append(Vhat)
            Rk += dR
            R.append(Rk)

        return ((pd.Series(R) - self.benchmark).dropna() ** 2).sum()



    def get_optimised_parameters(self):
        """
        loops through all sub Strategies, collects the data and finds the
        optimised dist_rate,dist_volatility,dist_correlation for the given dist of that strategy

        :return: list(dist_rate,dist_volatility,dist_correlation)
        """
        initial_guess = [0.08, 0.11, 0.8]
        bnds = ((0, 0.3), (0., 0.3), (0, 1.))
        _df = pd.DataFrame()
        for sub_strat in strategy_map[self.context.strategy]:
            substrat_df = datareader.Preqin.get_funds_filtered(category=sub_strat.lower(), vintage=('>', 2005),
                                                               status='closed',
                                                               size=('>', 200), region='', output='distribution_cum')
            substrat_df = (substrat_df[substrat_df.ffill() > 0].dropna(axis=1, how='all') / 10000000).mean(
                axis=1).fillna(0).ffill()
            _df = pd.concat([_df, substrat_df], axis=1)

        self.benchmark = _df.mean(axis=1)    #distributions are negative in preqin
        res = minimize(self.residuals_for_optimization, initial_guess, bounds=bnds)
        print('calculated optimised dist params for %s' %self.context.strategy)
        return res.x
