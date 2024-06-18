import os
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import joblib  
from pathlib import Path 

# mlflow 
from urllib.parse import urlparse 
import mlflow 
import mlflow.sklearn

# Set the environment variables for DagsHub authentication
os.environ['MLFLOW_TRACKING_USERNAME'] = 'minich-code'
os.environ['MLFLOW_TRACKING_PASSWORD'] = 'cadc5e14617d7fae5ed8a6532906afca14f3b0f9'

from src.RULBattery.entity.config_entity import ModelEvaluationConfig
from src.RULBattery.utils.commons import save_json


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config


    def predictions(self, model, X_test_transformed):
        
        y_pred = model.predict(X_test_transformed)
        
        return y_pred


    def model_evaluation(self, y_test, y_pred):
               
        # Evaluation metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

        
        # return
        return y_pred, mae, mse, r2, rmse, mape
        

    def save_results(self, y_test, y_pred):
        returned_values = self.model_evaluation(y_test, y_pred)
        mae, mse, r2, rmse, mape = returned_values[1:]  # Exclude y_pred

        # Saving metrics as local
        scores = {"MAE": mae, "MSE":mse, "R2": r2, "RMSE": rmse, "MAPE": mape}
        save_json(path=Path(self.config.metric_file_name), data=scores)


    def log_into_mlflow(self):

        # load the trained model, transformed test data and y_test data 
        model = joblib.load(self.config.model_path)
        X_test_transformed = joblib.load(self.config.test_data_path)

        y_test_df = pd.read_csv(self.config.test_target_variable)
        y_test = y_test_df[self.config.target_column]

        # make predictions 
        y_pred = self.predictions(model, X_test_transformed)

        # Evaluate and log metrics 
        _, mae, mse, r2, rmse, mape = self.model_evaluation(y_test, y_pred)  # Ensure returning correct values

        # mlflow setup 
        mlflow.set_tracking_uri(self.config.mlflow_uri) # setting the tracking uri
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme # get tracking URL for registry 

        # End any existing run 
        mlflow.end_run()

        with mlflow.start_run():
            # log parameters 
            mlflow.log_params(self.config.all_params) 

            # log metrics 
            mlflow.log_metric("MAE", mae)
            mlflow.log_metric("MSE", mse)
            mlflow.log_metric("R2", r2)
            mlflow.log_metric("RMSE", rmse)
            mlflow.log_metric("MAPE", mape)


            # log model for model registry 
            if tracking_url_type_store !="file":
                mlflow.sklearn.log_model(model, "model", registered_model_name= "XGBRegressor")

            else:
                mlflow.sklearn.log_model(model, "model")