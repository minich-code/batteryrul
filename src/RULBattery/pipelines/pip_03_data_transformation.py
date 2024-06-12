from pathlib import Path
from src.RULBattery import logging

from src.RULBattery.config.configuration import ConfigurationManager
from src.RULBattery.components.c_03_data_transformation import DataTransformation


PIPELINE_NAME = "DATA TRANSFORMATION PIPELINE"

class DataTransformationPipeline:
    def __init__(self):
        pass

    def run(self):
        try:
            with open(Path("artifacts\data_validation\status.txt"), "r") as f:
                # Read both lines from the status file
                status_line1 = f.readline().strip()
                status_line2 = f.readline().strip()

                # Extract the status from both lines
                status1 = status_line1.split(": ")[-1]
                status2 = status_line2.split(": ")[-1]

                # Check if both statuses are True
                if status1 == "True" and status2 == "True":
                    logging.info(f"The data validation pipeline has already been executed Successfully !!!!")
                    
                    logging.info(f"#====================== {PIPELINE_NAME} Started ================================#")
                    
                    config = ConfigurationManager()
                    data_transformation_config = config.get_data_transformation_config()
                    data_transformation = DataTransformation(config = data_transformation_config)
                    X_train, X_test, y_train, y_test = data_transformation.train_test_splitting()
                    train_data_path, test_data_path, y_train_data_path, y_test_data_path, preprocessor_path= \
                        data_transformation.initiate_data_transformation(X_train, X_test, y_train, y_test)
                    
                    # Store data
                    data_transformation.save_data(train_data_path, test_data_path, \
                                                   y_train_data_path, y_test_data_path, preprocessor_path)
                
                else:
                    raise Exception ("The Data Schema is Invalid")
                
                logging.info(f"# =========== {PIPELINE_NAME} Terminated Successfully ! ===========\n\nx*************x")
                

        except Exception as e:
            print(e)

