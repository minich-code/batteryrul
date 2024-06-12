import os 
import pandas as pd 
import numpy as np 
import joblib 
from scipy.stats import zscore

from src.RULBattery import logging 
from src.RULBattery.utils.commons import save_object
from src.RULBattery.entity.config_entity import DataTransformationConfig

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler





# Create a class to handle the actual data transformation process

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            logging.info("Outlier removal process has started")

            # Select numeric columns from the DataFrame
            numeric_columns = self.config.numerical_cols

            # Calculating For columns individually 
            for col in numeric_columns:
                
                z_scores = np.abs(zscore(df[col]))
                df = df[(z_scores < 3.0)]  # Filter rows based on individual column z-scores
            
            logging.info(f"Original number of rows: {df.shape[0]}")
            logging.info(f"Number of rows after filtering: {df.shape[0]}")

            logging.info("Outlier removal process has completed")
            
            return df
        
        except Exception as e:
            raise e
        
    def get_transformer_obj(self):
        try:
            logging.info("Data transformation process has started")

            # Select numeric and categorical columns
            numeric_columns = self.config.numerical_cols
            categorical_columns = self.config.categorical_cols

            # Define the pipeline and the preprocessing steps for the numeric columns
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', MinMaxScaler())
            ])
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])

            # Define the column transformer
            preprocessor = ColumnTransformer(transformers=[
                ('num', numeric_transformer, numeric_columns),
                ('cat', categorical_transformer, categorical_columns)
                ], 
            remainder='passthrough')
            
        
            logging.info("Data transformation process has completed")
            
            return preprocessor
        
        except Exception as e:
            raise e
        
    # Split the data into training and testing 
    def train_test_splitting(self):
        try:
            logging.info("Data Splitting process has started")

            df = pd.read_csv(self.config.data_path)
            df = self.remove_outliers(df)  # Remove outliers before splitting

            X = df.drop(columns=["rul"])
            y = df["rul"]

            logging.info("Splitting data into training and testing sets")

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            logging.info("Saving the training and testing data in artifacts")

            # Save the target variable for both training and testing
            y_train.to_csv(os.path.join(self.config.root_dir, "y_train.csv"), index=False)
            y_test.to_csv(os.path.join(self.config.root_dir, "y_test.csv"), index=False)

            logging.info("Data Splitting process has completed")

            return X_train, X_test, y_train, y_test

        except Exception as e:
            raise e


    # Initiate data transformation 
    def initiate_data_transformation(self, X_train, X_test, y_train, y_test):
        try:
            logging.info("Data transformation process has started")

            # Get the preprocessor object 
            preprocessor_obj = self.get_transformer_obj()

            # Transform the training and test data
            X_train_transformed = preprocessor_obj.fit_transform(X_train)
            X_test_transformed = preprocessor_obj.transform(X_test)

            # Save the preprocessing obj 
            preprocessor_path = os.path.join(self.config.root_dir, "preprocessor_obj.joblib")
            save_object(obj=preprocessor_obj, file_path=preprocessor_path)

            # Save the transformed data as sparse matrix
            X_train_transformed_path = os.path.join(self.config.root_dir, "X_train_transformed.joblib")
            X_test_transformed_path = os.path.join(self.config.root_dir, "X_test_transformed.joblib")
            joblib.dump(X_train_transformed, X_train_transformed_path)
            joblib.dump(X_test_transformed, X_test_transformed_path)

            logging.info("Data Transformation process has completed")

            # Return
            return X_train_transformed, X_test_transformed, y_train, y_test, preprocessor_path

        except Exception as e:
            raise e