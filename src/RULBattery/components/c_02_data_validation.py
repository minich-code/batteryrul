from src.RULBattery.entity.config_entity import DataValidationConfig

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
    