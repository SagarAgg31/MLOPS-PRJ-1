import os
import pandas as pd
import joblib
from sklearn.model_selection import RandomizedSearchCV
import lightgbm as lgb
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score
from src.custom_exception import CustomException
from src.logger import get_logger
from config.paths_config import *
from utils.common_functions import read_yaml, load_data
from scipy.stats import uniform, randint
from config.model_params import *
import mlflow
import mlflow.sklearn


## Intialize the logger
logger = get_logger(__name__)

class ModelTrainer:
    def __init__(self,train_path:str,test_path:str,model_output_path:str):
        self.train_path = train_path
        self.test_path = test_path
        self.model_output_path = model_output_path

        self.params_dist = LIGHTGM_PARAMS
        self.random_search_params = RANDOM_SEARCH_PARAMS

    def load_and_split_data(self):
        try:
            logger.info(f"Loading data from {self.train_path}")
            train_df = load_data(self.train_path)
            
            logger.info(f"Data loaded successfully from {self.train_path}")
            test_df = load_data(self.test_path)

            X_train = train_df.drop(columns=['booking_status'])
            y_train = train_df['booking_status']

            X_test = test_df.drop(columns=['booking_status'])
            y_test = test_df['booking_status']

            logger.info("Data Splitted Successfully for Model Training")

            return X_train,y_train,X_test,y_test
            
        except Exception as e:
            logger.error(f"Error in loading and splitting data {e}")
            raise CustomException("Failed to load data" ,e)
        
    def train_lgbm(self,X_train,y_train):
        try:
            logger.info("Intializing our model")

            lgbm_model = lgb.LGBMClassifier(random_state=self.random_search_params['random_state'])
            logger.info("Starting Randomized Search CV (HyperParameter Tunning)")

            random_search = RandomizedSearchCV(
                estimator=lgbm_model,
                param_distributions=self.params_dist,
                n_iter=self.random_search_params['n_iter'],
                cv = self.random_search_params['cv'],
                n_jobs = self.random_search_params['n_jobs'],
                verbose = self.random_search_params['verbose'],
                random_state = self.random_search_params['random_state'],
                scoring= self.random_search_params['scoring']
            )
            logger.info("HyperParameter Tuning started")

            random_search.fit(X_train,y_train)

            logger.info("HyperParameter Tuning completed successfully")
            best_params = random_search.best_params_
            best_lgbm_model = random_search.best_estimator_

            logger.info(f"Best parameters are: {best_params}")

            return best_lgbm_model
        
        except Exception as e:
            logger.error(f"Error in training the model {e}")
            raise CustomException("Failed to train model",e)
        
    def evaluate_model(self,model,X_test,y_test):
        try:
            logger.info("Evaluating our model")

            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test,y_pred)
            precision = precision_score(y_test,y_pred)
            recall = recall_score(y_test,y_pred)
            f1 = f1_score(y_test,y_pred)

            logger.info(f"Model Evaluation completed successfully")
            logger.info(f"Accuracy: {accuracy}")
            logger.info(f"Precision: {precision}")
            logger.info(f"Recall Score: {recall}")
            logger.info(f"F1 Score: {f1}")

            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }

        except Exception as e:
            logger.error(f"Error in evaluating the model {e}")
            raise CustomException("Failed to evaluate model",e)
        
    def save_model(self,model):
        try:
            os.makedirs(os.path.dirname(self.model_output_path),exist_ok=True)   
            joblib.dump(model,self.model_output_path) 
            logger.info(f"Model saved successfully at {self.model_output_path}")
        except Exception as e:
            logger.error(f"Failed to save the model {e}")
            raise CustomException("Failed to save model",e)
        
    def run(self):
        try:
            with mlflow.start_run():
                logger.info("Starting the Model Training Pipeline")
                logger.info("Starting our mlflow experimentation")
                logger.info("Logging the traning and testing dataset to mlflow")
                mlflow.log_artifact(self.train_path,artifact_path="datasets")
                mlflow.log_artifact(self.test_path,artifact_path="datasets")
                
                X_train,y_train,X_test,y_test = self.load_and_split_data()
                best_lgmb_model = self.train_lgbm(X_train,y_train)
                metrics = self.evaluate_model(best_lgmb_model,X_test,y_test)
                self.save_model(best_lgmb_model)
                logger.info("Logging the model into MLFLOW")
                mlflow.log_artifact(self.model_output_path,artifact_path="models")

                logger.info("Logging Params and Metrics to MLFLOW")
                mlflow.log_params(best_lgmb_model.get_params())
                mlflow.log_metrics(metrics)
                logger.info("Model Training Pipeline completed successfully")
        
        except Exception as e:
            logger.error(f"Error in Model Training Pipeline {e}")
            raise CustomException("Failed to run model training pipeline",e)
        
if __name__ == "__main__":
    train_path = PROCESSED_TRAIN_DATA_PATH
    test_path = PROCESSED_TEST_DATA_PATH
    model_output_path = MODEL_OUTPUT_PATH

    model_trainer = ModelTrainer(train_path,test_path,model_output_path)
    model_trainer.run()