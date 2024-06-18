# Import libraries 
from dataclasses import dataclass 
from pathlib import Path

# Data Ingestion Entity 

@dataclass
class DataIngestionConfig:
    root_dir: Path
    mongo_uri: str
    database_name: str
    collection_name: str

# Data Validation Entity
@dataclass
class DataValidationConfig:
    root_dir:Path
    STATUS_FILE: str
    data_dir: Path
    all_schema: dict 


# Data Transformation Entity 
@dataclass
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    numerical_cols: list
    categorical_cols: list


# Model trainer Entity 
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


# Model evaluation 
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
