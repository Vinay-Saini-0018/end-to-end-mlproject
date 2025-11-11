import os
import sys

from src.logger import logging
from src.exception import CustomException
from src.utils import evaluate_model,save_obj
from dataclasses import dataclass

from catboost import CatBoostRegressor 
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor)
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

from sklearn.metrics import r2_score

@dataclass
class ModelTrainConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")     #path of the trained model

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("splitting the training and testing data into x and y forms")
            x_train,y_train,x_test,y_test = (
                train_array[:,:-1],     #all rows, all column except last 
                train_array[:,-1],      #all rows, last column
                test_array[:,:-1],
                test_array[:,-1]          
            )

            models = {
                "Random forest":RandomForestRegressor(),
                "Gradient boosting":GradientBoostingRegressor(),
                "Ada boost": AdaBoostRegressor(),
                "Linear regression": LinearRegression(),
                "Decision tree": DecisionTreeRegressor(),
                "XGBregressor": XGBRegressor(),
                "Catboost Regressor": CatBoostRegressor(verbose=False),
                "KNeighbors Regressor": KNeighborsRegressor()
            }

            # here i am not using the parameters 

            # evaluate_models return the dictionary with accuary score and this will store in model_report (which also store dict) variable
            model_report:dict = evaluate_model(x_train,y_train,x_test,y_test,models)

            #finding the best model
            best_model_score = max(sorted(model_report.values()))

            #finding best model name from dict
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            logging.info(f"best model found:{best_model_name}")

            save_obj(
                    file_path = self.model_trainer_config.trained_model_file_path,
                    obj = best_model
            )

            predicted = best_model.predict(x_test)
            r2_square = r2_score(y_test,predicted)
            return r2_square


        except Exception as e:
            raise CustomException(e,sys)