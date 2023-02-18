import numpy as np
import pandas as pd
import requests
import datetime as dt
import re
from bs4 import BeautifulSoup
import pymongo
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
    results = soup.find_all("li", class_ = "propertyPrice")
    findings = int(re.findall(r'\d+', soup.find_all("h2")[0].text)[0])

    return results, findings

def athome_scrpr():

    today = dt.datetime.today()
    page_size = 20

    uris = ["https://www.athome.lu/srp/?tr=buy&q=bb769e8c&loc=L4-nord&ptypes=house", "https://www.athome.lu/srp/?tr=buy&q=6cbd09fa&ptypes=house", "https://www.athome.lu/srp/?tr=buy&q=faee1a4a&ptypes=house"]
    #base_uri = "https://www.athome.lu/srp/?tr=buy&q=6cbd09fa&ptypes=house"

    for u in uris:
        prices = {}
        price_lst = []

        base_uri = u
        if "L4" in base_uri:
            coll = "NORD"
        elif "6cbd09fa" in base_uri:
            coll = "WILTZ"
        else:
            coll = "Luxembourg"

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

        prices['data'] = price_lst
        prices['uploadDate'] = today

        pymongo.MongoClient(mongo_uri)[db][coll].insert_one(prices)

    return prices

def build_index(coll):

    average_index = {}

    for k in pymongo.MongoClient(mongo_uri)[db][coll].find().sort('uploadDate', 1):

        average = np.average(k['data'])
        median = np.median(k['data'])
        average_index[k['uploadDate']] = median

    series = pd.Series(average_index)

    plt.plot(series)
    plt.show()

if __name__ == '__main__':

    #athome_scrpr()
    build_index(coll = "WILTZ")