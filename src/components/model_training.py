import pandas as pd
import numpy as np
import os
import sys
from dataclasses import dataclass
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import GridSearchCV
from src.utils import evaluate_model,save_object
from src.logger import logging
from src.exception import CustomException


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            x_train, y_train = train_array[:, :-1], train_array[:, -1]
            x_test, y_test = test_array[:, :-1], test_array[:, -1]

            # Lists to store results
            model_list = []
            best_model_list = []
            r2_list_train = []
            r2_list_test = []
            rmse_list_train = []
            rmse_list_test = []

            # Model and parameters dictionary
            models = {
                "LinearReg": LinearRegression(),
                "LassoReg": Lasso(),
                "RidgeReg": Ridge(),
                "ElasticNet": ElasticNet(),
                "DecisionTree": DecisionTreeRegressor(),
                "RandomForest": RandomForestRegressor(),
                "XGBReg": XGBRegressor(),
                "CatBoost": CatBoostRegressor(verbose=False),
                "AdaBoostReg": AdaBoostRegressor()
            }

            params = {
                "LinearReg": {},  # No hyperparameter tuning for LinearRegression
                "LassoReg": {'alpha': [0.1, 1, 10, 100, 1000]},
                "RidgeReg": {'alpha': [0.1, 1, 10, 100, 1000]},
                "ElasticNet": {
                    'alpha': [0.1, 1, 10, 100, 1000],
                    'l1_ratio': [0.1, 0.5, 0.7, 0.9, 1.0]
                },
                "DecisionTree": {
                    "criterion": ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    "splitter": ['best', 'random'],
                    "max_features": ['sqrt', 'log2']
                },
                "RandomForest": {'n_estimators': [10, 20, 50, 100, 200]},
                "XGBReg": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'n_estimators': [10, 20, 50, 100, 200]
                },
                "CatBoost": {
                    'depth': [5, 10, 15],
                    'learning_rate': [0.1, 0.05, 0.01, 0.001],
                    'iterations': [10, 20, 50, 100]
                },
                "AdaBoostReg": {
                    'n_estimators': [10, 20, 50, 100, 200],
                    'learning_rate': [0.1, 0.05, 0.01, 0.001]
                }
            }

            # Loop through each model and perform GridSearchCV
            for model_name, model in models.items():
                param_grid = params[model_name]
                gs = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=1)
                gs.fit(x_train, y_train)

                best_model = gs.best_estimator_

                # Predictions
                y_train_pred = best_model.predict(x_train)
                y_test_pred = best_model.predict(x_test)

                # Evaluate using the evaluate_model function from utils
                model_train_mae, model_train_rmse, model_train_r2 = evaluate_model(y_train, y_train_pred)
                model_test_mae, model_test_rmse, model_test_r2 = evaluate_model(y_test, y_test_pred)

                model_list.append(model_name)
                best_model_list.append(best_model)
                r2_list_train.append(model_train_r2)
                r2_list_test.append(model_test_r2)
                rmse_list_train.append(model_train_rmse)
                rmse_list_test.append(model_test_rmse)

            # Create DataFrame to show results
            model_output = pd.DataFrame({
                "Model": model_list,
                "Train_R2_score": r2_list_train,
                "Test_R2_score": r2_list_test,
                "Train_rmse_score": rmse_list_train,
                "Test_rmse_score": rmse_list_test,
                "Best_Model": best_model_list
            })

                        

            # Sort the DataFrame based on the 'Test_R2_score' to get the best model
            best_model_row = model_output.loc[model_output['Test_R2_score'].idxmax()]
            best_model_name = best_model_row['Model']
            best_model = best_model_row['Best_Model']
            if best_model_row.Test_R2_score<0.6:
                raise CustomException("No Best Model Found")
            logging.info("Best model found")
            save_object(file_path=self.model_trainer_config.trained_model_file_path,obj=best_model)
            
            logging.info("model.pkl is saved in artifacts")
            # Make predictions using the best model
            y_best_pred = best_model.predict(x_test)

            # Print the best model's performance
            print(f"Best model: {best_model_name}:{r2_score(y_pred=y_best_pred, y_true=y_test)}")

            # Calculate the difference in RMSE between train and test
            model_output['diff'] = model_output.Test_rmse_score - model_output.Train_rmse_score
            model_output.sort_values(by='Test_R2_score', ascending=False, inplace=True)
            model_output_drop = model_output.drop('Best_Model', axis=1)
            logging.info("model training is completed")
            return model_output_drop 

        except Exception as e:
            raise CustomException(e, sys)
