# Import libraries 
from dataclasses import dataclass 
from pathlib import Path
import pymongo
from pymongo import MongoClient 
import pandas as pd
import os 

from src.RULBattery.utils.commons import read_yaml, create_directories
from src.RULBattery.constants import *
from src.RULBattery import logging 

# Entity 

@dataclass
class DataIngestionConfig:
    root_dir: Path
    mongo_uri: str
    database_name: str
    collection_name: str


# Creating a ConfigurationManager class to manage configurations
class ConfigurationManager:
    def __init__(
        self,
        config_filepath=CONFIG_FILE_PATH,
        params_filepath=PARAMS_FILE_PATH,
        schema_filepath=SCHEMA_FILE_PATH):

        """Initialize ConfigurationManager."""
        # Read YAML configuration files to initialize configuration parameters
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        # Create necessary directories specified in the configuration
        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """Get data ingestion configuration."""
        # Get data ingestion section from config
        config = self.config.data_ingestion

        # Creating the DataIngestionConfig object
        create_directories([config.root_dir])

        # Create and return DataIngestionConfig object
        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            mongo_uri=config.mongo_uri,
            database_name=config.database_name,
            collection_name=config.collection_name,
        )

        return data_ingestion_config

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


# Update the data_ingestion.py file
if __name__=="__main__":
    try:
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.import_data_from_mongodb()
        logging.info("Data Ingestion from MongoDB Completed!")
    except Exception as e:
        raise e