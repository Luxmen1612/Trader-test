import numpy as np
import pandas as pd
import numpy_financial as npf
from pyclausie import ClausIE
from numpy import random
import matplotlib.pyplot as plt
import os, pymupdf
import nltk


def pyclausie(jarfile):

    s = "The Sub-Fund is an actively managed, equity portfolio that invests, on a global basis (including up to 20% of its net assets in emerging markets), in global health care companies with the potential to benefit from dominant brand, market share, innovative technologies, or secular trends impacting the health care industry."
    cl = ClausIE.get_instance(jar_filename=jarfile)
    triples = cl.extract_triples([s])
    x = "test"


class FootballSim:
    def __init__(self, start = 8, period = 15, club_number = 2):

        self.df = {}
        self.start = start
        self.club_number = club_number
        self.period = period
        self.dr = 0.1

        self.academy_start = 200
        self.academy_cotisation = 350
        self.sports_pack = 150
        self.academy_growth = 0.05
        self.manfee = 50000
        self.sportswear_margin = 0.65
        self.sportswear_share = 0.75
        self.group_sponsoring = 25000

        self.manco_cfs = {}
        self.manco_npvs = {}
        self.sims = 100000
        self.legacy_cst = 0
        self.otpt = self.sim()

    def sim(self):

        self.league = {}
        self.df = pd.DataFrame()
        self.npvs = []
        df = {}

        for sim in range(self.sims):
            print(sim)
            df[sim], self.league[sim] = self.manco(sim)

            self.npvs.append(npf.npv(self.dr, df[sim]))
            self.df = pd.concat([self.df, pd.Series(df[sim])], axis = 1)


        self.median_index = self.npvs.index(pd.Series(self.npvs).median())
        self.median_cf = df[self.median_index]
        self.median_league = self.league[self.median_index]
        self.scenarios = {"unfavorable": pd.Series(self.npvs).quantile(0.1),
                          "median": pd.Series(self.npvs).median(),
                          "favorable": pd.Series(self.npvs).quantile(0.9)}

        test = ""

    def manco(self, sim):

        df = {}
        league = []
        league_dic = {}

        for k in range(self.club_number):
            cfs = []
            for i in range(self.period):
                if i == 0:
                    l = self.start

                else:
                    l = self.simulate_league(l)

                league.append(l)

                footco_profit, manfee, group_sponsoring = self.footco(i, l)
                academy, damra, academy_consumption = self.academy(i, l)
                employee_cost = 0 if i < 3 else 50000
                management_cost = 30000

                cfs.append(footco_profit - self.legacy_cst + manfee + damra + group_sponsoring - employee_cost - management_cost)

            final_value = (cfs[-1] / (self.dr*2))
            cfs[-1] = cfs[-1] + final_value
            df[k] = cfs

        if self.club_number == 2:
            agg_cf = pd.Series(df[0]) + pd.Series(df[1])

        elif self.club_number == 3:
            agg_cf = pd.Series(df[0]) + pd.Series(df[1]) + pd.Series(df[2])

        elif self.club_number == 1:
            agg_cf = pd.Series(df[0])

        else:
            agg_cf = pd.Series(df[0]) + pd.Series(df[1]) + pd.Series(df[2]) + pd.Series(df[3]) + pd.Series(df[4])


            #df[k] = (npf.npv(self.dr, pd.Series(cfs)) + final_value, l)

        return agg_cf, league


    def footco(self, i, l):
        """

        :param i: period
        :param l: league
        :return:
        """

        self.revenue_map = {
            8: {"attendance": [50, 5, 10, 40 * 150], "tv_rev": 0, "sponsoring": 20000, "other": 30000,
                "academy_size": 220, "group_sponsoring_share": 20000},
            7: {"attendance": [75, 5, 12, 50 * 150], "tv_rev": 0, "sponsoring": 30000, "other": 30000,
                "academy_size": 240, "group_sponsoring_share": 20000},
            6: {"attendance": [120, 10, 14, 60 * 150], "tv_rev": 0, "sponsoring": 35000, "other": 30000,
                "academy_size": 280, "group_sponsoring_share": 25000},
            5: {"attendance": [400, 12.5, 20, 120 * 150], "tv_rev": 0, "sponsoring": 40000, "other": 40000,
                "academy_size": 320, "group_sponsoring_share": 40000},
            4: {"attendance": [650, 15, 20, 150 * 150], "tv_rev": 0, "sponsoring": 65000, "other": 50000,
                "academy_size": 340, "group_sponsoring_share": 40000},
            3: {"attendance": [850, 15, 20, 180 * 250], "tv_rev": 0, "sponsoring": 90000, "other": 50000,
                "academy_size": 380, "group_sponsoring_share": 45000},
            2: {"attendance": [1000, 15, 25, 220 * 250], "tv_rev": 500000, "sponsoring": 110000, "other": 60000,
                "academy_size": 420, "group_sponsoring_share": 60000},
            1: {"attendance": [2000, 20, 30, 300 * 250], "tv_rev": 3000000, "sponsoring": 250000, "other": 90000,
                "academy_size": 500, "group_sponsoring_share": 100000}
        }

        self.cost_map = {
            8: {"team": 40000, "coaching_budget": 30000, "other": 30000, "academy_staff": 60000},
            7: {"team": 60000, "coaching_budget": 30000, "other": 30000, "academy_staff": 60000},
            6: {"team": 80000, "coaching_budget": 30000, "other": 30000, "academy_staff": 60000},
            5: {"team": 150000, "coaching_budget": 42000, "other": 30000, "academy_staff": 90000},
            4: {"team": 150000, "coaching_budget": 45000, "other": 30000, "academy_staff": 90000},
            3: {"team": 225000, "coaching_budget": 45000, "other": 30000, "academy_staff": 90000},
            2: {"team": 475000, "coaching_budget": 70000, "other": 30000, "academy_staff": 120000},
            1: {"team": 1500000, "coaching_budget": 90000, "other": 30000, "academy_staff": 150000}
        }


        if i == 0:
            academy_cf, damra_cf, academy_consumption = self.academy(i, l)
            cfs, manfee, group_sponsoring = self.simulate_sport(l, manfee_tag=False)[0] + academy_cf + academy_consumption, self.simulate_sport(l, manfee_tag=False)[1], self.simulate_sport(l, manfee_tag=False)[2]

        else:
            academy_cf, damra_cf, academy_consumption = self.academy(i, l)
            if i < 2:
                cfs, manfee, group_sponsoring = self.simulate_sport(l, manfee_tag=False)[0] + academy_cf + academy_consumption, self.simulate_sport(l, manfee_tag=False)[1], self.simulate_sport(l, manfee_tag=False)[2]
            else:
                cfs, manfee, group_sponsoring = self.simulate_sport(l, manfee_tag=True)[0] + academy_cf + academy_consumption, self.simulate_sport(l, manfee_tag=True)[1], self.simulate_sport(l, manfee_tag=True)[2]

        return cfs, manfee, group_sponsoring


    def simulate_sport(self, league, manfee_tag):

        manfee = self.manfee if manfee_tag is True else 0
        attendance, ticket, consumption, vip = self.revenue_map[league].get("attendance")
        revenues = attendance * ticket * 15 + attendance * consumption * 15 + vip + self.revenue_map[league].get("tv_rev") + self.revenue_map[league].get("sponsoring") + self.revenue_map[league].get("other")

        cost = self.cost_map[league].get("team") + manfee + self.cost_map[league].get("coaching_budget") + self.cost_map[league].get("other") + self.cost_map[league].get("academy_staff")

        group_sponsoring_share = self.revenue_map[league].get("group_sponsoring_share")

        return revenues-cost, manfee, group_sponsoring_share


    def simulate_league(self, start):

        promotion_probs = {5: [-1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                           8: [-1, 0, 0, 0]}

        probs = 8 if start > 5 else 5
        l_incr = random.choice(promotion_probs[probs])
        end = l_incr + start
        end = 1 if end < 1 else end
        end = 8 if end > 8 else end

        return end

    def academy(self, i, league):

        academy_size = self.revenue_map[league].get("academy_size")
        academy_revenues = academy_size * self.academy_cotisation * (1+0.02)**(i)
        academy_cost = academy_size * self.sports_pack * (1+0.02)**(i)
        academy_consumption = academy_size * (10 * (1+0.035)**(i)) * 15

        damra_cf = academy_size * self.sports_pack* (1+0.02)**i * self.sportswear_margin * self.sportswear_share

        return academy_revenues-academy_cost, damra_cf, academy_consumption


    def combine_and_plot(self):

        df = pd.DataFrame()

        npv_sum_lst = []
        npv_median = []
        median_league_lst = []

        for k,v in self.df.items():
            npv_sum = 0
            league_lst = []
            for x,y in v.items():
                npv_sum += y[0]
                league_lst.append(y[1])

            npv_sum_lst.append(npv_sum)
            npv_median.append(np.median(npv_sum))
            median_league_lst.append(np.median(league_lst))

        return pd.concat([df, pd.Series(npv_sum_lst), pd.Series(npv_median), pd.Series(median_league_lst)], axis = 1)



def simulate_league(start, period = 15, club_number = 5) :

    promotion_probs = {5: [-1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                       8: [0, 0, 0, -1]}

    for c in range(club_number):
        for x in range(1):
            start_base = start
            league_lst = []
            for i in range(period):
                if i == 0:
                    end = 7
                else:
                    probs = 8 if start_base > 5 else 5
                    l_incr = random.choice(promotion_probs[probs])
                    end = l_incr + start_base
                    end = 1 if end < 1 else end
                    end = 8 if end > 8 else end

                start_base = end
                league_lst.append(end)

            plt.plot(league_lst)

    plt.xlabel("Year")
    plt.ylabel("League")
    plt.show()


def plot_league_revenues():

    revenue_map = {
        8: {"attendance": [50, 5, 10, 40 * 150], "tv_rev": 0, "sponsoring": 20000, "other": 30000,
            "academy_size": 220, "group_sponsoring_share": 20000, "academy_cost": 350},
        7: {"attendance": [75, 5, 12, 50 * 150], "tv_rev": 0, "sponsoring": 30000, "other": 30000,
            "academy_size": 240, "group_sponsoring_share": 20000, "academy_cost": 350},
        6: {"attendance": [120, 10, 14, 60 * 150], "tv_rev": 0, "sponsoring": 35000, "other": 30000,
            "academy_size": 280, "group_sponsoring_share": 25000, "academy_cost": 385},
        5: {"attendance": [400, 12.5, 20, 140 * 150], "tv_rev": 0, "sponsoring": 40000, "other": 40000,
            "academy_size": 320, "group_sponsoring_share": 40000, "academy_cost": 385},
        4: {"attendance": [650, 15, 20, 150 * 150], "tv_rev": 0, "sponsoring": 65000, "other": 50000,
            "academy_size": 340, "group_sponsoring_share": 40000, "academy_cost": 400},
        3: {"attendance": [850, 15, 20, 180 * 250], "tv_rev": 0, "sponsoring": 90000, "other": 50000,
            "academy_size": 380, "group_sponsoring_share": 45000, "academy_cost": 420},
        2: {"attendance": [1000, 15, 25, 220 * 250], "tv_rev": 500000, "sponsoring": 110000, "other": 60000,
            "academy_size": 420, "group_sponsoring_share": 60000, "academy_cost": 450},
        1: {"attendance": [2000, 20, 30, 300 * 250], "tv_rev": 3000000, "sponsoring": 250000, "other": 90000,
            "academy_size": 500, "group_sponsoring_share": 100000, "academy_cost": 450}
    }

    cost_map = {
        8: {"team": 40000, "coaching_budget": 30000, "other": 30000, "academy_staff": 60000},
        7: {"team": 60000, "coaching_budget": 30000, "other": 30000, "academy_staff": 60000},
        6: {"team": 80000, "coaching_budget": 30000, "other": 30000, "academy_staff": 60000},
        5: {"team": 150000, "coaching_budget": 42000, "other": 30000, "academy_staff": 90000},
        4: {"team": 150000, "coaching_budget": 45000, "other": 30000, "academy_staff": 90000},
        3: {"team": 225000, "coaching_budget": 45000, "other": 30000, "academy_staff": 90000},
        2: {"team": 475000, "coaching_budget": 70000, "other": 30000, "academy_staff": 120000},
        1: {"team": 1500000, "coaching_budget": 90000, "other": 30000, "academy_staff": 150000}
    }

    rev_lst = []
    cost_lst = []

    for i in range(8):

        l = i + 1

        attendance, ticket, consumption, vip = revenue_map[l].get("attendance")

        revenue = attendance * (ticket + consumption) + vip + revenue_map[l].get("tv_rev") + revenue_map[l].get("sponsoring") + revenue_map[l].get("other") + \
                  revenue_map[l].get("academy_size") * (revenue_map[l].get("academy_cost")-150) + revenue_map[l].get("academy_size") * 12.5 * 15

        cost = cost_map[l].get("team") + cost_map[l].get("coaching_budget") + cost_map[l].get("other") + cost_map[l].get("academy_staff")

        rev_lst.append(revenue)
        cost_lst.append(cost)

        plt.plot(rev_lst)
        plt.plot(cost_lst)
        plt.xlabel("League")
        plt.ylabel("Revenue / cost")
        plt.ticklabel_format(style='plain')
    plt.show()




from PIL import Image
import pytesseract

def ocr():
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract'
    file = "G:\.shortcut-targets-by-id\\1Q4IY1-fdN6Fd4F87_Dok01_O3zmSIPtr\FundRisQ\Clients\\15 - LEIW AIFM\Directorship Agreement\\20231215_115406.jpg"
    doc = pytesseract.image_to_string(Image.open(file))

    return doc


class transformer:

    def __init__(self):

        self.base_uri = "G:\\.shortcut-targets-by-id\\1Q4IY1-fdN6Fd4F87_Dok01_O3zmSIPtr\\FundRisQ\\FRQ-NLP\\documents"
        self.tokens = []

    def read_directory(self):

        for f in os.listdir(self.base_uri):
            uri = f"G:\.shortcut-targets-by-id\\1Q4IY1-fdN6Fd4F87_Dok01_O3zmSIPtr\FundRisQ\FRQ-NLP\documents\{f}"

            if f in ["Samples Prospectus", "prospectus"]:
                for sf in os.listdir(uri):
                    try:
                        self.tokens.extend(self.read_doc(uri + f"\{sf}"))

                    except:
                        continue

            elif f == "desktop.ini":
                continue

            else:
                self.tokens.extend(self.read_doc(uri))


    def read_doc(self, uri):

        content = ""
        doc = pymupdf.open(uri)
        for page in doc: # iterate the document pages
            content += page.get_text().encode("utf8").decode()

        return nltk.word_tokenize(content)


if __name__ == "__main__":

    t = transformer()
    t.read_directory()
    test = 1

    ###################### FOOTBALL GPV ###########################
    #npv = {}
    #percentiles = [0.1, 0.9]

    #for c in [1]:
    #    f = FootballSim(6, 10, c)
    #    npv[c] = {"mean": f.scenarios["median"], "unfavorable": f.scenarios["unfavorable"], "favorable": f.scenarios["favorable"]}

    ####################################################

    #example_investment = 400000
    #annual_yield = 400000*0.1

    #plt.plot(f.median_cf)
    #plt.plot(pd.Series(np.zeros(len(f.median_cf)) + annual_yield))
    #plt.ticklabel_format(style='plain')
    #plt.show()

    #simulate_league(7, 15, 5)
    #plot_league_revenues()



