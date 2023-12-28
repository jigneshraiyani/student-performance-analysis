import os, sys
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from src.utils import save_object, evaluate_model

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor)
from xgboost import XGBRegressor

from sklearn.metrics import r2_score

FOLDER_NAME = 'artifacts'

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join(FOLDER_NAME, 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.trained_model_config = ModelTrainerConfig()

    def initiate_model_training(self, train_arr, test_arr):
        try:
            logging.info("Split training and test input data")
            x_train, y_train, x_test, y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            models = {
                'Linear Regression': LinearRegression(),
                'KNeighbors Regressor': KNeighborsRegressor(),
                'DecisionTree Regressor': DecisionTreeRegressor(),
                'RandomForest Regressor': RandomForestRegressor(),
                'GradientBoostingRegressor': GradientBoostingRegressor(),
                'AdaBoostRegressor': AdaBoostRegressor(),
                'XGB Regressor': XGBRegressor()
            }

            model_report = evaluate_model(x_train=x_train, 
                                          y_train=y_train,
                                          x_test=x_test,
                                          y_test=y_test,
                                          models=models)
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(models.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.trained_model_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(x_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square
        except Exception as e:
            raise CustomException(e,sys)

