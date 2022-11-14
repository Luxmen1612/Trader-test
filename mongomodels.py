import pymongo
from pymongo import MongoClient

class MongoClient:
    def __init__(self, uri, db_name, db_coll):

        self.uri = uri
        self.db_name = db_name
        self.db_coll = db_coll

        self.client = pymongo.MongoClient(uri)

        self.db()

    def db(self):

        self.db = self.client[self.db_name][self.db_coll]

    def insert(self, data):

        self.db.insert_one(data)

    def delete(self, query):

        self.db.delete_one(query)

    def find_one(self, query):

        data = self.db.find_one(query)
        return data
