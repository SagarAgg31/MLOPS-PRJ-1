from src.custom_exception import CustomException
from src.logger import get_logger
import os
import pandas as pd
import numpy as np
from config.paths_config import *
from utils.common_functions import read_yaml, load_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

logger = get_logger(__name__)

class DataProcessor:
    def __init__(self,train_path:str,test_path:str,processed_dir:str,config_path:str):
        self.train_path = train_path
        self.test_path = test_path
        self.processed_dir = processed_dir

        self.config_path = read_yaml(config_path)

        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir,exist_ok=True)
            logger.info(f"Directory {self.processed_dir} created successfully")
        
    def preprocess_data(self,df):
        try:
            logger.info('Starting our Data Processing steps!')

            logger.info("Dropping the columns")
            df.drop(['Booking_ID'], axis=1, inplace=True)
            logger.info("Dropping the duplicates")
            df.drop_duplicates(inplace=True)

            data_preprocessing_config = self.config_path['data_processing']
            cat_cols = data_preprocessing_config['categorical_cols']
            num_cols = data_preprocessing_config['numerical_cols']

            logger.info('Applying Label Encoding')
            label_encoder = LabelEncoder()
            mapping = {}
            for col in cat_cols:
                df[col] = label_encoder.fit_transform(df[col])
                mapping[col] = {label:code for label,code in zip(label_encoder.classes_,label_encoder.transform(label_encoder.classes_))}

            logger.info('Label Mappings are: ')
            for col,mapping in mapping.items():
                logger.info(f'{col}: {mapping}')

            logger.info('Doing Skewness Handling')
            skewness_threshold = data_preprocessing_config['skewness_threshold']
            for col in num_cols:
                if df[col].skew() > skewness_threshold:
                    df[col] = np.log1p(df[col])
                    logger.info(f"Applied log transformation on {col}")

            return df
        
        except Exception as e:
            logger.error(f"Error occurred in data preprocessing {e}")
            raise CustomException('Error while preprocess data',e)
    
    def balance_data(self,df):
        try:
            logger.info('Starting our Data Balancing steps!')
            X = df.drop(columns='booking_status')
            y =df['booking_status']
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X,y)
            balanced_df = pd.DataFrame(X_resampled,columns=X.columns)
            balanced_df['booking_status'] = y_resampled
            logger.info('Data balancing completed!')
            return balanced_df

        except Exception as e:
            logger.error(f"Error occurred in data balancing {e}")
            raise CustomException('Error while balancing data',e)
        
    def feature_selection(self,df):
        try:
            logger.info('Starting our Feature Selection steps!')
            X = df.drop(columns='booking_status')
            y = df['booking_status']

            rf = RandomForestClassifier(random_state=42)
            rf.fit(X, y)

            feature_importances = rf.feature_importances_
            feature_importances_df = pd.DataFrame({'feature': X.columns, 'importance': feature_importances})
            top_features_df =feature_importances_df.sort_values(by='importance',ascending=False)
            num_feature_to_select = self.config_path['data_processing']['no_of_features']
            top_10_features = feature_importances_df['feature'].head(num_feature_to_select).values
            top_10_df = df[top_10_features.tolist() + ['booking_status']]
            logger.info(f"Top {num_feature_to_select} features selected: {top_10_features}")
            logger.info('Feature selection completed!')
            return top_10_df

            
        except Exception as e:
            logger.error(f"Error occurred in feature selection {e}")
            raise CustomException('Error while selecting features',e)
        
    def save_data(self,df,file_path:str):
        try:
            logger.info('Saving our data in processed folder')
            df.to_csv(file_path,index=False)
            logger.info(f"Data saved successfully at {file_path}")
        except Exception as e:
            logger.error(f"Error occurred in saved_data {e}")
            raise CustomException('Error while saving the data',e)
        
    def DataPreprocessingProcess(self):
        try:
            logger.info("Loading data from RAW Directory")

            train_df = load_data(self.train_path)
            test_df = load_data(self.test_path)
            # Preprocess data
            train_df = self.preprocess_data(train_df)
            test_df = self.preprocess_data(test_df)
            # Balance data
            train_df = self.balance_data(train_df)
            test_df = self.balance_data(test_df)
            # Feature selection
            train_df = self.feature_selection(train_df)
            test_df = test_df[train_df.columns]

            # Save data
            self.save_data(train_df,PROCESSED_TRAIN_DATA_PATH)
            self.save_data(test_df,PROCESSED_TEST_DATA_PATH)
            logger.info("Data Preprocessing completed successfully!")

        except Exception as e:
            logger.error(f"Error occurred in Data Preprocessing {e}")
            raise CustomException('Error in Data Preprocessing Process',e)
        

if __name__ == "__main__":
    try:
        data_processor = DataProcessor(TRAIN_FILE_PATH,TEST_FILE_PATH,PROCESSED_DIR,CONFIG_PATH)
        data_processor.DataPreprocessingProcess()
    except Exception as e:
        logger.error(f"Error occurred in main {e}")
        raise CustomException('Error in main function',e)


        


