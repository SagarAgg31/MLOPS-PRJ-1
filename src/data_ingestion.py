import os
import pandas as pd
from google.cloud import storage
from sklearn.model_selection import train_test_split
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.common_functions import read_yaml
import sys

logger = get_logger(__name__)

class DataIngestion:
    def __init__(self,config):
        self.config = config['data_ingestion']
        self.bucket_name = self.config['bucket_name']
        self.file_name = self.config['bucket_file_name']
        self.train_test_ratio = self.config['train_ratio']

        os.makedirs(RAW_DIR,exist_ok=True)
        logger.info(f"Data Ingestestion started with {self.bucket_name} and file is {self.file_name}")

    def download_csv_from_gcp(self):
        try:
            client = storage.Client()
            print(client)
            bucket = client.bucket(self.bucket_name)
            print(bucket)
            blob = bucket.blob(self.file_name) # means filename
            blob.download_to_filename(RAW_FILE_PATH)
            logger.info("File downloaded from GCP successfully!")
        except Exception as e:
            logger.error("Error occurred while downloading the file from GCP")
            raise CustomException(e,sys) from e
    def train_test_split(self):
        try:
            logger.info("Starting the Splitting process")
            data = pd.read_csv(RAW_FILE_PATH)
            logger.info("The data has been read successfully")
            train_data,test_data = train_test_split(data,train_size=self.train_test_ratio,random_state=42)
            logger.info("The data has been splitted into train and test")
            train_data.to_csv(TRAIN_FILE_PATH,index=False)
            test_data.to_csv(TEST_FILE_PATH,index=False)
            logger.info(f"Train data is saved to {TRAIN_FILE_PATH}")
            logger.info(f"Test data is saved to {TEST_FILE_PATH}")
            return train_data,test_data
        except Exception as e:
            logger.error("Error occurred while splitting the data into train and test")
            raise CustomException(e,sys) from e
        
    def run(self):
        try:
            logger.info("Starting Data Ingestion process")
            self.download_csv_from_gcp()
            self.train_test_split()
            logger.info("Data Ingestion completed successfully!")
        except Exception as e:
            logger.error("Error occurred in Data Ingestion")
            raise CustomException(e,sys) from e
        
if __name__ == "__main__":
    try:
        config = read_yaml(CONFIG_PATH)
        data_ingestion = DataIngestion(config)
        data_ingestion.run()
    except Exception as e:
        logger.error("Error occurred in Data Ingestion")
        raise CustomException(e,sys) from e
