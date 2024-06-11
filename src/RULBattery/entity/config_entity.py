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
