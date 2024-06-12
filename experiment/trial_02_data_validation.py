# Importing necessary libraries to handle entities and configurations
from dataclasses import dataclass 
from pathlib import Path 

# Importing specific constants and utility functions from custom modules
from src.RULBattery.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH, SCHEMA_FILE_PATH
from src.RULBattery.utils.commons import read_yaml, create_directories

# Importing component related libraries 
import os 
from src.RULBattery import logging
import pandas as pd 


# Defining the structure of data validation configuration using a data class
@dataclass
class DataValidationConfig:
    root_dir:Path
    STATUS_FILE: str
    data_dir: Path
    all_schema: dict 

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

    
    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation 
        schema = self.schema.COLUMNS

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            STATUS_FILE=config.STATUS_FILE,
            data_dir=config.data_dir,
            all_schema=schema
        )
        return data_validation_config
    
class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config =config

    
    def validate_all_columns(self, data):
        try:
            validation_status = True
            all_cols = list(data.columns)
            all_schema = list(self.config.all_schema.keys())

            missing_columns = [col for col in all_schema if col not in all_cols]
            extra_columns = [col for col in all_cols if col not in all_schema]

            if missing_columns or extra_columns:
                validation_status = False

            with open(self.config.STATUS_FILE, 'w') as f:
                f.write(f"Validation status: {validation_status}\n")
                if missing_columns:
                    f.write(f"Missing columns: {missing_columns}\n")
                if extra_columns:
                    f.write(f"Extra columns: {extra_columns}\n")

            return validation_status
        
        except Exception as e:
            raise e
        
    # Method to validate the data types of all columns of a dataset against a specified schema
    # Method to validate data types of all columns
    def validate_data_types(self, data):
        try:
            validation_status = True
            all_schema = self.config.all_schema

            type_mismatches = {}
            for col, expected_type in all_schema.items():
                if col in data.columns:
                    actual_type = data[col].dtype
                    if actual_type != expected_type:
                        type_mismatches[col] = (expected_type, actual_type)
                        validation_status = False

            with open(self.config.STATUS_FILE, 'a') as f:
                f.write(f"Data type validation status: {validation_status}\n")
                if type_mismatches:
                    f.write(f"Type mismatches: {type_mismatches}\n")

            return validation_status

        except Exception as e:
            raise e
    
        

## Main pipeline
if __name__=="__main__":
    try:
        # Initialize ConfigurationManager to get configuration settings
        config = ConfigurationManager()
        # Get data validation configuration
        data_validation_config = config.get_data_validation_config()
        # Create DataValidation object with the obtained configuration
        data_validation = DataValidation(config=data_validation_config)

        # Fetch data from CSV (or alternatively directly from MongoDB)
        data = pd.read_csv(data_validation_config.data_dir)
        
        # Perform data validation for all columns
        column_validation_status = data_validation.validate_all_columns(data)
        
        # Perform data type validation for all columns
        type_validation_status = data_validation.validate_data_types(data)

        if column_validation_status and type_validation_status:
            logging.info("Data Validation Completed Successfully!")
        else:
            logging.info("Data Validation Failed. Check the status file for more details.")

    except Exception as e:
        raise e
