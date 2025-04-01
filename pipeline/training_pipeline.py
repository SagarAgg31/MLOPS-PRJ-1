from src.logger import logging
from src.custom_exception import CustomException
from utils.common_functions import *
from config.model_params import *
from config.paths_config import *
from src.data_ingestion import DataIngestion
from src.data_preprocessing import DataProcessor
from src.model_training import ModelTrainer

if __name__ == "__main__":
    ## 1. Data Ingestion
    logging.info("Pipeline started")
    config = read_yaml(CONFIG_PATH)
    data_ingestion = DataIngestion(config)
    data_ingestion.run()
    logging.info("Data Ingestion completed")
    ## 2. Data Preprocessing and Data Validation
    logging.info("Data Preprocessing started")
    data_processor = DataProcessor(TRAIN_FILE_PATH,TEST_FILE_PATH,PROCESSED_DIR,CONFIG_PATH)
    data_processor.DataPreprocessingProcess()
    logging.info("Data Preprocessing completed")
    ## 3. Model Training And Evaluation
    logging.info("Model Training started")
    train_path = PROCESSED_TRAIN_DATA_PATH
    test_path = PROCESSED_TEST_DATA_PATH
    model_output_path = MODEL_OUTPUT_PATH

    model_trainer = ModelTrainer(train_path,test_path,model_output_path)
    model_trainer.run()
    logging.info("Model Training completed")

