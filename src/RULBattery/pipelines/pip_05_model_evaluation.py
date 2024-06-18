from src.RULBattery.config.configuration import ConfigurationManager
from src.RULBattery.components.c_05_model_evaluation import ModelEvaluation
from src.RULBattery import logging 
import joblib 
import pandas as pd 
import mlflow 


PIPELINE_NAME = "MODEL EVALUATION PIPELINE"


class ModelEvaluationPipeline:
    def __init__(self):
        pass 


    def run(self):
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()

        # Create ModelEvaluation object
        model_evaluation = ModelEvaluation(config=model_evaluation_config)

        # End any existing run
        mlflow.end_run() 

        # Log to MLflow
        model_evaluation.log_into_mlflow()


if __name__=="__main__":
    try:
        logging.info(f"# ============== {PIPELINE_NAME} Started ================#")
        model_evaluation_pipeline = ModelEvaluationPipeline()
        model_evaluation_pipeline.run()
        logging.info(f"# ============= {PIPELINE_NAME} Terminated Successfully ! ===========\n\nx******************x") 
    except Exception as e:
        logging.exception(e)
        raise e