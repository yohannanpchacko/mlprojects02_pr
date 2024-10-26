import pandas as pd
import numpy as np
import os
import sys
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.utils import save_object

@dataclass
class data_transformation_config:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")
class DataTransformation:
    def __init__(self):
        self.data_tranformation_config=data_transformation_config()
    def get_data_transformer_obj(self):
        try:
            numeric_columns=["writing_score","reading_score"]
            categorical_columns=['gender','race_ethnicity','parental_level_of_education','lunch','test_preparation_course']
            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='median')),
                    ("scalar",StandardScaler())
                ]
            )
            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='most_frequent')),
                    ("onehotencoding",OneHotEncoder()),
                    ("scalar",StandardScaler(with_mean=False))
                ]
            )
            logging.info(f"numeric columns: {numeric_columns}")
            logging.info(f"categorical columns: {categorical_columns}")
            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numeric_columns),
                ("cat_pipleline",cat_pipeline,categorical_columns)
                ]
            )
            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("read train test data is completed")
            logging.info("obtaining preprocessor object")
            preprocessor_obj=self.get_data_transformer_obj()
            target_col='math_score'
            input_feature_train_df=train_df.drop(columns=[target_col],axis=1)
            target_feature_train_df=train_df[target_col]
            input_feature_test_df=test_df.drop(columns=[target_col],axis=1)
            target_feature_test_df=test_df[target_col]
            logging.info("applying preprocessing obj on train and test data frame")
            input_feature_train_arry=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_array=preprocessor_obj.transform(input_feature_test_df)
            train_arr=np.c_[input_feature_train_arry,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_array,np.array(target_feature_test_df)]
            save_object(
                file_path=self.data_tranformation_config.preprocessor_obj_file_path,obj=preprocessor_obj)


            return (
                train_arr,
                test_arr,
                self.data_tranformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)




