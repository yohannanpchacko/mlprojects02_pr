import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from src.components.data_transformation import DataTransformation,data_transformation_config
from src.components.model_training import ModelTrainerConfig, ModelTrainer
from src.utils import evaluate_model
@dataclass
class DataIngestionConfig:
    raw_data_file_path:str=os.path.join('artifacts','data.csv')
    train_data_file_path:str=os.path.join('artifacts','train.csv')
    test_data_file_path:str=os.path.join('artifacts','test.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    def initiate_data_ingestion(self):
        logging.info("Data Ingestion initiated")
        try:
            df=pd.read_csv('notebook\data\stud.csv')
            logging.info("data set reading is completed")
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_file_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_file_path,index=False,header=True)

            logging.info("train test split is initiated")
            train_set,test_set=train_test_split(df,test_size=0.20,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_file_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_file_path,index=False,header=True)
            logging.info("train test split is completed")
            return (self.ingestion_config.train_data_file_path,
                    self.ingestion_config.test_data_file_path)
        
        except Exception as e:
            raise CustomException(e,sys)
if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()
    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_path=train_data,test_path=test_data)
    model_trainer=ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr,test_arr))

        
