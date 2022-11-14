import alpaca_trade_api as alpaca
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime as dt

import toolbox
from toolbox import yield_calc, get_momentum, get_price
import multiprocessing as mp

#def import_data(symbol, dict):

#    data = yf.download(symbol)
#    dict[symbol] = data

#    return dict

class user:

    def __init__(self, uri = '/home/david/Documents/ticker_list.xlsx'):

        api_key = 'PKKKT5ZCJ0GFT8RW06L3'
        api_secret = 'cwjDkwzHdtrCiH2X1QMoNZTuTJiYAXnffkEkEBTS'
        base_url = 'https://paper-api.alpaca.markets'
        self.current_year = dt.today().year
        self.file = pd.read_excel(uri, engine='openpyxl')

        self.api = alpaca.REST(api_key, api_secret, base_url, api_version='v2')
        self.active_assets = self.api.list_assets(status = 'active')

        self.investment_amount = 75000
        self.symbol_list = list(set(self.file['SYMBOL']))

    def calibrate(self):

        div_spread = None
        dict_av_yield = {}
        dict_portfolio = {}
        div_dict = {}
        mom_dict = {}
        start = 2010
        end = self.current_year
        df = pd.Series()

        for i in range(start, end):
            for f in self.symbol_list:
                try:
                    _yield = yield_calc(f, i)
                    # _ret = get_momentum(f, i)
                    div_dict[f] = _yield
                    # mom_dict[f] = _ret

                except:
                    print('Data error occurred')

            div_universe = pd.Series(div_dict).sort_values(ascending=True)
            dict_av_yield[i] = np.average(div_universe)

            div_top_yield = np.percentile(div_universe, 95)
            div_low_yield = 0
            div_spread = div_top_yield - div_low_yield

            long_portion = list(div_universe[-100:].index.values)


            #short_universe = div_universe[div_universe <= dict_av_yield[i]]

            short_portion = list(pd.Series(div_dict).sort_values(ascending = True).index.values[:100])

            portfolio = long_portion + short_portion

            dict_portfolio[i] = portfolio

            ser = pd.Series(portfolio)
            ser.name = i

            df = pd.concat([df, ser], axis=1)

        df.to_excel("test.xlsx")

        return dict_portfolio, dict_av_yield, div_spread

    def test_performance(self):

        pass

    def div_series(self):

        year = 2022
        cut_off = - 1
        div_dict = {}
        div_global = {}
        start = -20

        for i in range(year + cut_off - 2000):
            for f in self.symbol_list:
                try:
                    _yield = yield_calc(f, (i+1))
                    div_dict[f] = _yield

                except:
                    print(f'{f} data issues')

            av_yield = np.average(pd.Series(div_dict))
            div_global[i] = av_yield

        return div_global


    def backtest(self):

        div_dict = {}
        mom_dict = {}
        self.portfolio = {}
        self.growth_ret = {}
        self.long_ret = {}
        self.premia = {}

        cut_off_date = '2021-12-31'
        start_date = '2015-12-31'
        #start_date = '2021-12-31'
        ref_dates = yf.download('^GSPC').resample('Y').last()[start_date: cut_off_date]

        # GET DIV YIELD #
        for i in ref_dates.index.values:
            for f in self.symbol_list:

                try:
                    _yield = yield_calc(f, i)
                    #_ret = get_momentum(f, i)
                    div_dict[f] = _yield
                    #mom_dict[f] = _ret

                except:
                    print('Data error occurred')

            div_universe = pd.Series(div_dict).sort_values(ascending = True)
            self.av_yield = np.average(div_universe)

            long_portion = div_universe[-50:].index.values
            short_universe = div_universe[div_universe <= self.av_yield]

            for f in short_universe.index.values:
                _ret = get_momentum(f, i)
                mom_dict[f] = _ret

            short_portion = pd.Series(mom_dict).sort_values(ascending = True).index.values[:50]
            self.portfolio[i] = (long_portion, short_portion)
            ret = self.back_test_portfolio(i, long_portion, short_portion)

            self.long_ret[i] = ret[0]
            self.growth_ret[i] = ret[1]

            self.premia[i] = self.long_ret[i] - self.growth_ret[i]

            #self.portfolio[i] = ret


    def back_test_portfolio(self, date, long_portion, short_portion):

        long_portion_ret = 1
        short_portion_ret = 1
        start_price = 0
        end_price = 0

        next_year = date + np.timedelta64(365, 'D')
        #
        if str(next_year)[9] != '1':
            next_year = next_year + np.timedelta64(1, 'D')
        elif str(next_year)[8] == '0':
            next_year = next_year - np.timedelta64(1, 'D')

        for i in long_portion:

            #long_ret = (get_price(i, next_year) /  get_price(i, date))
            #long_portion_ret = long_portion_ret * long_ret
            start_price += get_price(i, date)
            end_price += get_price(i, next_year)

        long_ret = (end_price / start_price) - 1

        for y in short_portion:

            #short_ret = (get_price(y, next_year) /  get_price(y, date))
            #short_portion_ret = short_portion_ret * short_ret
            start_price += get_price(y, date)
            end_price += get_price(y, next_year)

        short_ret = (end_price / start_price) - 1

        return long_ret, short_ret


    def filter_assets(self):

        #self.dict = mp.Manager().dict()
        self.dict = {}
        self.shortable_assets = self.file['SYMBOL']
        self.data = []

        #for a in self.active_assets:
        #    sec_data = a._raw
        #    sec_data_df = pd.DataFrame(sec_data, index=[0])
        #    if sec_data['easy_to_borrow'] is True and sec_data['shortable'] is True\
        #            and sec_data['marginable'] is True and sec_data['class'] != 'crypto':
        #
        #        self.shortable_assets.append(sec_data['symbol'])



        for s in self.shortable_assets:
            try:

                #data = yf.download(s)
                #self.data.append(data)
                _yield = yield_calc(s, self.year)
                self.dict[s] = _yield
                print('data has been successfully downloaded')

            except:
                print(f' {s} not in yfinance')

        self.yield_ser = pd.Series(self.dict)

        self.long_portion = self.yield_ser.sort_values(ascending = False)[:9]

        short_universe = self.yield_ser.sort_values(ascending = False)[-50:].index.values
        self.momentum = toolbox.get_momentum(short_universe, self.year)

        #portfolio = create_portfolio()



if __name__ == '__main__':

    user = user()
    user.calibrate()
    user.backtest()





