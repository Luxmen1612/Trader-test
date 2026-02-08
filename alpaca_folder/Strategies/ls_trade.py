from alpaca_folder.alpaca import get_data, order, generate_ticket
import pandas as pd
import numpy as np
from scipy.optimize import minimize

from toolbox import model, sum_of_squares


def pair_trade_ls(tickers, weight_func = "beta"):

    """

    :param tickers: 2 ticker symbols, first long, second short
    :param weight: weight function
    :return: order transaction
    """
    ticker_direction = {tickers[0]: "buy", tickers[1]: "sell"}

    weight = {}
    for k in tickers:
        if weight_func == "beta":
            weight[k] = beta_weight(k, BM = k)

    for k,v in weight.items():

        ticket = generate_ticket(k, np.round(v), ticker_direction[k])
        order(ticket)


def equiv_weight(tickers):

    weight = 1 / len(tickers)

    return weight


def beta_weight(ticker, BM):

    bm_ret = get_data(BM).pct_change().dropna()
    alpha = 0.
    beta = 0.

    ret = get_data(ticker).pct_change().dropna()
    mod = model([alpha, beta], bm_ret)
    res = minimize(sum_of_squares, [alpha, beta], args=(ret, bm_ret))


    return res.x[1]


if __name__ == "__main__":

    x = pair_trade_ls(["INTC", "AMD"], weight_func = "beta")
