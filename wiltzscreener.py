import numpy as np
import pandas as pd
import requests
import datetime as dt
import re
import pymongo
from bs4 import BeautifulSoup
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from dotenv import dotenv_values
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
config = dotenv_values(BASE_DIR/ ".env")

mongo_uri = config['MONGO_DB_URI']

db = "HOUSE_PRICES"

def get_content(uri):

    page = requests.get(uri)
    soup = BeautifulSoup(page.content, "html.parser")
    results = soup.find_all("span", class_ = "property-card-price")
    #findings = int(re.findall(r'\d+', soup.find_all("h2")[0].text))
    findings = int(re.findall(r'\d+', soup.find_all("h2")[0].text.replace(",", ""))[0])

    return results, findings

def athome_scrpr():

    today = dt.datetime.today()
    page_size = 20

    #uris = ["https://www.athome.lu/srp/?tr=buy&q=bb769e8c&ptypes=house",
    #    #"https://www.athome.lu/srp/?tr=buy&q=bb769e8c&loc=L4-nord&ptypes=house",
    #        "https://www.athome.lu/srp/?tr=buy&q=6cbd09fa&ptypes=house"]

    #uris = {"NORD": "https://www.athome.lu/srp/?tr=buy&q=bb769e8c&ptypes=house",
    uris = {"WILTZ": "https://www.athome.lu/srp/?tr=buy&q=6cbd09fa&ptypes=house"}

    for u,v in uris.items():
        coll = u
        prices = {}
        price_lst = []

        #base_uri = u
        base_uri = v
        #if "L4" in base_uri:
        #    coll = "NORD"
        #elif "6cbd09fa" in base_uri:
        #    coll = "WILTZ"
        #else:
        #    coll = "Luxembourg"

        results = get_content(base_uri)[0]
        findings = get_content(base_uri)[1]

        for r in results:
            price = re.sub(r'\s+','', r.text)
            price_lst.append(int(re.findall(r'\d+', price)[0]))

        if findings > page_size:
            pages = np.ceil(findings / page_size)

            for p in range(int(pages-1)):
                uri = base_uri + f"&page={p+2}"
                results = get_content(uri)[0]
                for r in results:
                    price = re.sub(r'\s+', '', r.text)
                    try:
                        price_lst.append(int(re.findall(r'\d+', price)[0]))

                    except:
                        pass

        prices['data'] = price_lst[:(findings-1)]
        prices['uploadDate'] = today

        pymongo.MongoClient(mongo_uri)[db][coll].insert_one(prices)

    return prices

def build_index(coll, data ="prices"):

    average_index = {}
    supply_index = {}

    for k in pymongo.MongoClient(mongo_uri)[db][coll].find().sort('uploadDate', 1):
        #if data == "price":
        average = np.average(k['data'])
        median = np.median(k['data'])
        average_index[k['uploadDate']] = median
            #supply = len(k["data"])
            #supply_index[k["uploadDate"]] = supply

        #else:
        supply = len(k["data"])
        supply_index[k["uploadDate"]] = supply

    series = pd.Series(average_index)
    #series_supply = pd.Series(supply_index)

    #plt.plot(series)
    #plt.plot(series_supply)
    #plt.show()

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5, 3))
    axes[0].plot(pd.Series(average_index))
    axes[1].plot(pd.Series(supply_index))
    fig.tight_layout()
    plt.show()


def model(params, X):
    # here you need to implement your real model
    # for Predicted_Installation

    alpha = params[0]
    a1 = params[1]
    a2 = params[2]
    a3 = params[3]
    a4 = params[4]
    a5 = params[5]
    a6 = params[6]
    a7 = params[7]

    #y_pred = alpha + a1 * X["Liveable space"] + a2 * X["Parcel size"] + a3 * X["Parking"] + a4 * X["New/Old"] + a5 * X["State"] + a6 * X["Free"] * a7 * X["Bedrooms"]
    y_pred = alpha + a1 * X["Liveable space"] + a2 * X["Parcel size"] + a3 * X["Parking"] + a4 * X["New/Old"] + a5 * X["State"] + a6 * X["Free"] * a7 * X["Bedrooms"]


    return y_pred

def sum_of_squares(params, X, Y):

    y_pred = model(params, X)
    obj = np.sqrt(((y_pred - Y) ** 2).sum())
    return obj

if __name__ == '__main__':

    athome_scrpr()

    for k in ["WILTZ", "NORD"]:
    #for k in ["NORD"]:
        build_index(coll = k, data = "price")

    #file = pd.read_excel("C:\\Users\\raths\OneDrive\Desktop\Wiltz_prices.xlsx", sheet_name = "Sheet2")
    #file = pd.read_excel("C:\\Users\\raths\PycharmProjects\pythonProject\Wiltz_prices.xlsx", sheet_name = "300325")
    #output = file["Price"]
    #file = file.drop(labels = ["Asset Nr", "Price"], axis = 1)
    #alpha, a1, a2, a3, a4, a5, a6, a7 = 0,0,0,0,0,0,0,0
    #input = file
    #mod = model([alpha, a1, a2, a3, a4, a5, a6, a7], input)
    #mod = model([a1, a2, a3, a4, a5, a6, a7], input)
    #res = minimize(sum_of_squares, [alpha, a1, a2, a3, a4, a5, a6, a7], args = (input, output))
    #asset_param = [128, 380, 4, -1, 3, 1, 5]
    #price = res.x[0] + asset_param[0] * res.x[1] + asset_param[1] * res.x[2] + asset_param[2] * res.x[3] + asset_param[3] * res.x[4] + asset_param[4] * res.x[5] + asset_param[5] * res.x[6] + asset_param[6] * res.x[7]
    #test = 1