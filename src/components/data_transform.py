import sys   #for errors
import os    #for paths
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.logger import logging
from src.exception import CustomException
from src.utils import save_obj

@dataclass
class DataTransformationConfig:     #class for the defining the path of the preprocessor.pkl file
    preprocessor_obj_file_path = os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:
    def __init__(self):   #creating variable which stores the address of pkl file of preprocessor
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:

            #defining the numerical and the categorical column of the dataset
            numerical_column = ["writing_score","reading_score"]
            categorical_column = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]

            #pipeline for categorical columns
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),    #this will fill the null value by most frequent values of that particular column
                    ("one_hot_encoder",OneHotEncoder()),    #this will convert categorical column into numerical
                    ("standardscaler",StandardScaler(with_mean=False))      #this will Standardize (put data into the same range )
                ]
            )

            #pipeline for numerical columns
            num_pipeline= Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),    #this will fill the null value by most frequent values of that particular column
                    ("scaler",StandardScaler())                    
                ]
            )

            logging.info(f"categorical columns : {categorical_column}")
            logging.info(f"numerical columns : {numerical_column}")

            #combining both the pipeline so that it work on complete dataframe
            preprocessor = ColumnTransformer([
                ("num_pipeline",num_pipeline,numerical_column),
                ("cat_pipeline",cat_pipeline,categorical_column)
            ])
            
            return preprocessor   #return the pipeline of preprocessing

        except Exception as e:
            raise CustomException(e,sys)

    #function to initiate the 
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("reading of train and test data is completed")
            logging.info("obtaining preprocessor pipeline")

            preprocessor_obj = self.get_data_transformer_object()       #preprocessor pipeline
            target_column_name = "math_score"
            numerical_columns = ["writing_score","reading_score"]

            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)     # x of training data
            target_feature_train_df = train_df[target_column_name]      # y of training data

            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)     # x of testing data
            target_feature_test_df = test_df[target_column_name]      # y of testing data        

            logging.info("applying preprocesssing object to training and testing dataframe")

            #preprocessing on input training and the testing dataframe
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            # train and test array by combining input and the target column
            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            logging.info("saving preprocesing obj as a pkl file")       # so that we can use it on another data

            # calling utils function to save the pkl file
            save_obj(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,     #preprocessor_obj_file_path is a variable which stores the path of folder in string format
                obj=preprocessor_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        
        except Exception as e:
            raise CustomException(e,sys)








