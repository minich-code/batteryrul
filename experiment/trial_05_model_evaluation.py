from pathlib import Path
from dataclasses import dataclass 

import dagshub
import mlflow 
dagshub.init(repo_owner='minich-code', repo_name='batteryrul', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/minich-code/batteryrul.mlflow")


@dataclass()
class ModelEvaluationConfig:
    root_dir: Path
    test_data_path: Path
    test_target_variable: Path
    model_path: Path
    all_params: dict
    metric_file_name: Path
    target_column: str
    # mlflow 
    mlflow_uri: str


from src.RULBattery.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH, SCHEMA_FILE_PATH
from src.RULBattery.utils.commons import read_yaml, create_directories, save_json

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH,
        schema_filepath = SCHEMA_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        create_directories([self.config.artifacts_root])


    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        params = self.params.XGBRegressor
        schema = self.schema.TARGET_COLUMN

        create_directories([config.root_dir])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=config.root_dir,
            test_data_path=config.test_data_path,
            test_target_variable= config.test_target_variable,
            model_path = config.model_path,
            all_params=params,
            metric_file_name = config.metric_file_name,
            target_column = schema.name,
            # mlflow
            mlflow_uri = "https://dagshub.com/minich-code/batteryrul.mlflow"

        )
        return model_evaluation_config


import os
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import joblib   

# mlflow 
from urllib.parse import urlparse 
import mlflow 
import mlflow.sklearn


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


# # Without model tracking 

# if __name__ == "__main__":
#     try:
#         config = ConfigurationManager()
#         model_evaluation_config = config.get_model_evaluation_config()

#         # Load the trained model, transformed test data, and y_test data
#         model = joblib.load(model_evaluation_config.model_path)
#         X_test_transformed = joblib.load(model_evaluation_config.test_data_path)

#         y_test_df = pd.read_csv(model_evaluation_config.test_target_variable)
#         y_test = y_test_df[model_evaluation_config.target_column]

#         # Create ModelEvaluation object
#         model_evaluation = ModelEvaluation(config=model_evaluation_config)

#         # Get predictions
#         y_pred = model_evaluation.predictions(model, X_test_transformed)


#         # Save evaluation results
#         evaluation_results = model_evaluation.save_results(y_test, y_pred)

#     except Exception as e:
#         raise e

# With model tracking 
if __name__ == "__main__":
    try:
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()

        # Create ModelEvaluation object
        model_evaluation = ModelEvaluation(config=model_evaluation_config)

        # End any existing run
        mlflow.end_run() 

        # Log to MLflow
        model_evaluation.log_into_mlflow()

    except Exception as e:
        raise e