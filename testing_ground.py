import numpy as np
import pandas as pd
import numpy_financial as npf

def cst_arr(amt, rhp, place = 'beg'):

    cost_arr = np.zeros(rhp)

    if place == 'beg':

        cost_arr[0] = amt
    else:
        cost_arr[-1] = amt

    return cost_arr

def calc_net_to_gross(net_cf, cost, rhp):

    if rhp == 1:
        cfs_net = [-10000, 10000]


def incr_irr_calc(net_cf, gross_cf, cost_dict, gross_irr, riy):

    ratios = {}

    for cost in cost_dict.keys():
        ratios[cost] = gross_irr - (
            npf.irr(np.array(gross_cf) - (cst_arr(cost_dict[cost] * 10000, len(net_cf), place='end'))))

    total_cost = np.sum(ratios.values())
    ratios['carry'] = max(riy - total_cost, 0)

    return ratios

def calc_carry(commitment, value, carry_pct):

    carry = commitment + ((value - commitment) / carry_pct) - value

    return carry

class Cost:

    def __int__(self):

        self.commitment = 10000

    def moonfare_feeder(self, mc_obj, nav_lst, perc, fund_dict, feeder_dict, rhp):

        cost_dict = {
            "entry": feeder_dict['entry'] / rhp,
            "exit": feeder_dict['exit'] / rhp,
            "ongoing": feeder_dict["ongoing"],
            "transaction" : feeder_dict["transaction"]
        }

        global_cost = np.sum(cost_dict.values())
        cost_amt = global_cost * self.commitment * rhp

        base_model = self.single_fund_model(mc_obj, nav_lst, perc, fund_dict)

        base_fund_riy = base_model[0]
        base_cost_riy = base_model[1] + cost_amt
        base_model_ratios = base_model[2]

        base_scenarios = np.percentile(nav_lst, [10, 50, 90])
        new_scenarios = base_scenarios + cost_amt
        moderate_scenario = new_scenarios[1]

        new_cost = self.single_fund_model(mc_obj, nav_lst, moderate_scenario, feeder_dict, rhp)

        riy = base_fund_riy + new_cost[0]
        new_cost_amt = base_cost_riy + new_cost[1]
        ratios = {}

        for k in feeder_dict.keys():
            ratios[k] = base_model_ratios[k] + new_cost[2][k]

        return riy, new_cost_amt, ratios, new_scenarios

    def single_fund_model(self, mc_obj, perc, nav_lst, dict, rhp): #for single fund and coinvest

        cost_dict = {
            "entry": dict['entry'] / rhp,
            "exit": dict['exit'] / rhp,
            "ongoing": dict["ongoing"],
            "transaction" : dict["transaction"]
        }

        moderate_scenario = perc

        carry = 0 if rhp == 1 else max(0 , self.commitment +
                                            ((moderate_scenario-self.commitment) /
                                            (1 - dict['carry']))
                                            - moderate_scenario)

        global_cost = np.sum(cost_dict.values())

        index_value = nav_lst.index(nav_lst[min(range(len(nav_lst)), key=lambda i: abs(nav_lst[i] - moderate_scenario))])

        if rhp == 1:
            net_irr = 0
            gross_cf = self.nominal + (carry + (global_cost * self.nominal * rhp))
            cfs_gross = pd.Series([-10000, gross_cf])
            gross_irr = npf.irr(np.array(cfs_gross))
            riy = gross_irr - net_irr

            cfs_net = pd.Series([-10000, 10000])

            ratios = incr_irr_calc(rhp, cfs_net, cost_dict, cfs_gross, riy)

            cost_amt = global_cost * self.nominal * rhp

        else:
            cfs = mc_obj[index_value].dC
            cfs.iloc[-1] += (mc_obj.V.iloc[-1] + (moderate_scenario - nav_lst[index_value]))
            cfs = cfs.reset_index().groupby(cfs.index // 4).sum()['dC']

            if len(cfs) > rhp:
                adj_val = np.sum(cfs[len(cfs) - rhp:])
                cfs = cfs[:rhp - 1]
                cfs.iloc[-1] = cfs.iloc[-1] + adj_val

            net_irr = npf.irr(np.array(cfs))

            cfs_gross = cfs  # + (global_cost * self.nominal)
            cfs_gross.iloc[-1] += (carry + (global_cost * self.nominal * rhp))
            gross_irr = npf.irr(np.array(cfs_gross))

            riy = gross_irr - net_irr

            ratios = incr_irr_calc(rhp, cfs, cost_dict, cfs_gross, riy)

            self.carry = carry

            cost_amt = (global_cost * self.nominal * rhp) + carry

        # return self.carry, self.cf, self.riy, self.cost_amt
        return riy, cost_amt, ratios

def perf_dynamic_table(document, cat_obj, lan = "en"):

    lan_map = map(cat_obj)
    col_len = None
    lan_mapper = {
        3: 1,
        4: 10,
        5: np.inf
    }

    for k,v in lan_mapper.items():
        if cat_obj.rhp < v:
            col_len = k
            break

    if cat_obj.cat == 1:

        col_len = 3

    type = ["stress", "stress_an", "unfavorable", "unfavorable_ann", "moderate", "moderate_ann", "favorable",
            "favorable_ann"]

    t_style = 'Light Grid Accent 1'
    rows = 11

    table = document.add_table(rows = rows, cols = col_len)

    for r in range(rows):
        if r < (rows - len(type)):
            if r == 0:
                cell = 0
                row = table.rows[r]
                b = row.cells[cell]
                cell += 1
                row.cells[cell].merge(b).text = lan_map.p_table_head.get(lan)

                for x in range(2, col_len):
                    row.cells[cell + 1].text(getattr(lan_map, f'p_table_{col_len-2}').get(lan))

            else:
                if r == 1:
                    row = table.rows[r]
                    b = row.cells
                else:
                    row = table.rows[r]
                    row.merge(b).text = "Minimum"



