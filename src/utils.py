import sys
import os

import numpy as np
import pandas as pd
import dill
import pickle

from src.logger import logging
from src.exception import CustomException

from sklearn.metrics import r2_score

def save_obj(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            pickle.dump(obj,file_obj)


    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_model(x_train,y_train,x_test,y_test,models):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]

            model.fit(x_train,y_train)      #training model
            y_train_pred = model.predict(x_train)       #finding pred of training data
            y_test_pred = model.predict(x_test)         #finding pred of testind data

            train_model_score = r2_score(y_train,y_train_pred)      #finding the accuracy on training data
            test_model_score = r2_score(y_test,y_test_pred)         #finding the accuracy on testing data

            report[list(models.keys())[i]] = test_model_score       #this will add test_model_score as value with model name as key

        return report


    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path,"rb") as file_obj:
           return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e,sys)