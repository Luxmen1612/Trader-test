from toolbox import _Brownian
from private_markets import jcurve as buchner_main
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy import stats


class Kupiec:

    def __init__(self, context, rate, pod=0., eod=1.,
                 lgd=0.5, seed = 0):

        self.context = context
        self.rate = rate
        self.pod = 0.2
        self.eod = 1
        self.lgd = 0.5
        self.indicator = {}
        self.seed = seed

        _, self.e = _Brownian(self.context.npoints, self.context.lifespan, self.seed)

    def update_Vk(self,obj): #original

        corr = 0.5
        _em = np.random.standard_normal()
        _ei = np.random.standard_normal()
        _v = np.sqrt(corr)*_em + np.sqrt(1-corr)*_ei
        self.indicator[obj.k] = 1 if _v < stats.norm.ppf(self.pod) else 0

        _residualV = obj.Vk * 1 if self.indicator[obj.k] == 0 else obj.Vk - obj.Vk * self.pod * self.lgd
        obj.Vk = _residualV * (1 + self.rate) + obj.dD - obj.dR
