from src.RULBattery import logging
from src.RULBattery.config.configuration import ConfigurationManager
from src.RULBattery.components.c_01_data_ingestion import DataIngestion


PIPELINE_NAME = "DATA INGESTION PIPELINE"

class DataIngestionPipeline:
    def __init__(self):
        pass 

    def run(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.import_data_from_mongodb()
        logging.info("Data Ingestion from MongoDB Completed!")


if __name__=="__main__":
    try:
        logging.info(f"## =================== Starting {PIPELINE_NAME} pipeline ========================##")
        data_ingestion_pipeline = DataIngestionPipeline()
        data_ingestion_pipeline.run()
        logging.info(f"## ================== {PIPELINE_NAME} Terminated Successfully!=======================\n\nx*****************x")
    except Exception as e:
        raise e 