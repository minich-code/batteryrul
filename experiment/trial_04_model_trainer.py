from dataclasses import dataclass
from pathlib import Path
import numpy as np
from typing import Literal
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Entity 
@dataclass()
class ModelTrainerConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    model_name: str
    # XGBOOST parameters 
    objective: str  
    booster: str
    n_estimators: int
    learning_rate: float
    max_depth: int
    min_child_weight: int
    gamma: float
    subsample: float
    colsample_bytree: float
    reg_alpha: float
    reg_lambda: float
    random_state: int
    scale_pos_weight: int


from src.RULBattery.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH, SCHEMA_FILE_PATH
from src.RULBattery.utils.commons import read_yaml, create_directories, save_json 
from experiment.trial_03_data_transformation import DataTransformationConfig

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

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=Path(config.root_dir),
            data_path=Path(config.data_path),
            numerical_cols=list(config.numerical_cols),
            categorical_cols=list(config.categorical_cols)
        )
        # Return 
        return data_transformation_config 
    
    
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params.XGBRegressor
        schema =  self.schema.TARGET_COLUMN
        
        
        create_directories([config.root_dir])


        model_trainer_config = ModelTrainerConfig(
            root_dir=config.root_dir,
            train_data_path = config.train_data_path,
            test_data_path = config.test_data_path,
            model_name = config.model_name,
            # XGBOOST parameters 
            objective=params['objective'],
            booster=params['booster'],
            n_estimators=params['n_estimators'],
            learning_rate=params['learning_rate'],
            max_depth=params['max_depth'],
            min_child_weight=params['min_child_weight'],
            gamma=params['gamma'],
            subsample=params['subsample'],
            colsample_bytree=params['colsample_bytree'],
            reg_alpha=params['reg_alpha'],
            reg_lambda=params['reg_lambda'],
            random_state=params['random_state'],
            scale_pos_weight=params['scale_pos_weight'],
            

        )
        return model_trainer_config

import pandas as pd 
import os 
from src.RULBattery import logging
from xgboost import XGBRegressor
import joblib 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from experiment.trial_03_data_transformation import DataTransformation


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
    
    def initiate_model_trainer(self, X_train_transformed, y_train):

        xgb_model = XGBRegressor(
            objective=self.config.objective,
            booster=self.config.booster,
            n_estimators=self.config.n_estimators,
            learning_rate=self.config.learning_rate,
            max_depth=self.config.max_depth,
            min_child_weight=self.config.min_child_weight,
            gamma=self.config.gamma,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            reg_alpha=self.config.reg_alpha,
            reg_lambda=self.config.reg_lambda,
            random_state=self.config.random_state,
            scale_pos_weight=self.config.scale_pos_weight,
        )

        xgb_model.fit(X_train_transformed, y_train)
        
        joblib.dump(xgb_model, os.path.join(self.config.root_dir, self.config.model_name))
        logging.info(f"Model saved to {os.path.join(self.config.root_dir, self.config.model_name)}") 


if __name__ == "__main__":
    try:
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        data_transformation_config = config.get_data_transformation_config()  

        data_transformation = DataTransformation(config=data_transformation_config)
        X_train, X_test, y_train, y_test = data_transformation.train_test_splitting()

        # Transform the data
        X_train_transformed, X_test_transformed, _, _, _ = data_transformation.initiate_data_transformation(
            X_train, X_test, y_train, y_test
        )

        model_trainer = ModelTrainer(config=model_trainer_config)
        model_trainer.initiate_model_trainer(X_train_transformed, y_train)

    except Exception as e:
        raise e
        