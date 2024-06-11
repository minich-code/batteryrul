import pymongo
from pymongo import MongoClient 
import pandas as pd
import os 

from src.RULBattery.utils.commons import read_yaml, create_directories
from src.RULBattery.constants import *
from src.RULBattery import logging 
from src.RULBattery.entity.config_entity import DataIngestionConfig


# Data ingestion component class 
class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    # Method to fetch data from MongoDB
    def import_data_from_mongodb(self):
        # Connect to MongoDB
        client = pymongo.MongoClient(self.config.mongo_uri)
        db = client[self.config.database_name]
        collection = db[self.config.collection_name]

        # Convert the collection to a DataFrame
        df = pd.DataFrame(list(collection.find()))

        if "_id" in df.columns:
            df = df.drop(columns=["_id"])


        # Save DataFrame to a file (optional, based on your needs)
        df.to_csv(os.path.join(self.config.root_dir, 'battery_rul.csv'), index=False)