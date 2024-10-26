import os
import numpy as np
import pandas as pd
import sys
import dill
from src.exception import CustomException
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from sklearn.model_selection import GridSearchCV

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)
def evaluate_model(actual,predicted):
    mae=mean_absolute_error(actual,predicted)
    mse=mean_squared_error(actual,predicted)
    rmse=np.sqrt(mse)
    R2_score=r2_score(actual,predicted)
    return mae,rmse,R2_score
def load_obj(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e,sys)
