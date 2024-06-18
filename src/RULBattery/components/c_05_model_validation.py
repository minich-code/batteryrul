import numpy as np 
import pandas as pd 
import os 
import sys 

from src.RULBattery import logging
from src.RULBattery.exception import FileOperationError
from src.RULBattery.utils.commons import load_object


# Create a class for the prediction pipeline 
class PredictionPipeline:
    def __init__(self):
        pass 

    def make_predictions(self, features):
        try:
            logging.info("Making predictions")

            # Define the model and preprocessor_obj 
            model_path = os.path.join("artifacts", "model_trainer", "model.joblib")
            preprocessor_path = os.path.join("artifacts", "data_transformation", "preprocessor_obj.joblib")
            
            # Load the preprocessor and the model 
            preprocessor_obj = load_object(file_path=preprocessor_obj)
            model = load_object(file_path=model_path)

            # Transform the features 
            features_transformed = preprocessor_obj.transform(features)

            # Make predictions 
            predictions = model.predict(features_transformed)

            # Return the predictions 
            return predictions
        
        except Exception as e:
            raise FileOperationError (e, sys)