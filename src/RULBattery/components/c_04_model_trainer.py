import pandas as pd 
import os 
from src.RULBattery import logging
from xgboost import XGBRegressor
import joblib 
from src.RULBattery.entity.config_entity import ModelTrainerConfig


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