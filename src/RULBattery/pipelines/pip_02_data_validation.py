from src.RULBattery.config.configuration import ConfigurationManager
from src.RULBattery.components.c_02_data_validation import DataValidation
from src.RULBattery import logging 
import pandas as pd 


PIPELINE_NAME = "DATA VALIDATION PIPELINE"

class DataValidationPipeline:
    def __init__(self):
        pass 


    def run(self):
   
        config = ConfigurationManager()
     
        data_validation_config = config.get_data_validation_config()
     
        data_validation = DataValidation(config=data_validation_config)

    
        data = pd.read_csv(data_validation_config.data_dir)
     
        column_validation_status = data_validation.validate_all_columns(data)
      
        type_validation_status = data_validation.validate_data_types(data)

        if column_validation_status and type_validation_status:
            logging.info("Data Validation Completed Successfully!")
        else:
            logging.info("Data Validation Failed. Check the status file for more details.")

if __name__ =="__main__":
    try:
        logging.info(f"# ================ {PIPELINE_NAME} Started ================#")
        data_validation_pipeline = DataValidationPipeline()
        data_validation_pipeline.run()
        logging.info(f"# ================= {PIPELINE_NAME} Terminated Successfully ! ===============\n\nx*******************x")
    except Exception as e:
        print(e)