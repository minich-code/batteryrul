from src.RULBattery.utils.commons import read_yaml, create_directories
from src.RULBattery.constants import *
from src.RULBattery.entity.config_entity import DataIngestionConfig



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

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """Get data ingestion configuration."""
        # Get data ingestion section from config
        config = self.config.data_ingestion

        # Create DataIngestionConfig object
        create_directories([config.root_dir])

        # Create and return DataIngestionConfig object
        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            mongo_uri=config.mongo_uri,
            database_name=config.database_name,
            collection_name=config.collection_name,
        )

        return data_ingestion_config