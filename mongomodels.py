import pymongo
from pymongo import MongoClient

from dotenv import dotenv_values
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
config = dotenv_values(BASE_DIR/ ".env")

mongo_uri = config["MONGO_DB_URI"]

class MongoClient:

    def __init__(self, db_name, db_coll):

        self.uri = mongo_uri
        self.db_name = db_name
        self.db_coll = db_coll

        self.client = pymongo.MongoClient(self.uri)

        self.db()

    def db(self):

        self.db = self.client[self.db_name][self.db_coll]

    def find(self, query, sort_by_date = True, sort_type = "ascending", limit = 1):

        if sort_by_date is True and sort_type == "ascending":
            cursor = self.db.find(query).sort('uploadDate', 1).limit(limit)

        elif sort_by_date is True and sort_type == "descending":
            cursor = self.db.find(query).sort('uploadDate', -1).limit(limit)

        else:
            cursor = self.db.find(query)

        return cursor

    def insert_many(self, data):

        self.db.insert_many(data)

    def delete(self, query):

        self.db.delete_one(query)

    def find_one(self, query):

        data = self.db.find_one(query)
        return data
