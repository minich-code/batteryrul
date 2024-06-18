# batteryrul

# Sequence 
``
 - constants - logger, exception etc 
 - update the config 
 - schema 
 - params 
 - entity 
 - config in config manager 
 - components 
 - Pipeline 
 - main.py 
 - prediction pipeline 

``

# 1. Entity: DataIngestionConfig
``
What it is: This is a data class that defines the structure for data ingestion configuration. It acts like a blueprint for how the ingestion process will be set up.

What it does: It holds configuration details required for data ingestion, such as the root directory, MongoDB URI, database name, and collection name.

Input: Configuration details read from a YAML configuration file.

Output: An instance of DataIngestionConfig with the specified configuration details.

``

# 2. Configuration: ConfigurationManager
``
What it is: A class that manages configurations by reading them from YAML files.

What it does:
    - Reads configuration, parameters, and schema details from YAML files.
    - Creates necessary directories specified in the configuration.
    - Provides a method to get the data ingestion configuration as an instance of DataIngestionConfig.

Input: Paths to configuration, parameters, and schema YAML files.
    - config_filepath: The path to the YAML file containing the general project configuration.
    - params_filepath: The path to the YAML file containing the parameters for the data ingestion process.
    - schema_filepath: The path to the YAML file defining the schema for the data.

Output: Configuration details and a method to get the data ingestion configuration (get_data_ingestion_config)

Usage: You'd instantiate ConfigurationManager to load your YAML files and then call get_data_ingestion_config to obtain the specific configuration for data ingestion.

``
#  Data Ingestion: DataIngestion

``
What it is: A class responsible for the actual data ingestion process. Fetching data from a MongoDB database and saving it to a CSV file.

What it does: Connects to MongoDB, fetches data from a specified collection, converts it to a DataFrame, and saves it as a CSV file.

Input: An instance of DataIngestionConfig containing the necessary configuration details.
    - A DataFrame containing the data from MongoDB (potentially written to a CSV file)

Output: A CSV file containing the data fetched from MongoDB.

Usage: You'd create an instance of DataIngestion, passing in the data_ingestion_config obtained from ConfigurationManager. Then, you would call import_data_from_mongodb to perform the ingestion task.

``

# Pipeline 

``
What it is: The main execution block of the script. The data ingestion process when the script is run directly

What it does:
    - Initializes the configuration manager.
    - Retrieves the data ingestion configuration.
    - Creates an instance of DataIngestion with the configuration (data_ingestion_config).
    - Invokes the data ingestion process to fetch data from MongoDB.

Input: None directly, but it uses the configuration files and the classes defined above.

Output: Executes the data ingestion process and logs the completion status. The imported data from MongoDB (possibly saved to a CSV file).
``

``
export MLFLOW_TRACKING_URI=https://dagshub.com/minich-code/batteryrul.mlflow
export MLFLOW_TRACKING_USERNAME=minich-code
export MLFLOW_TRACKING_PASSWORD=cadc5e14617d7fae5ed8a6532906afca14f3b0f9
``
