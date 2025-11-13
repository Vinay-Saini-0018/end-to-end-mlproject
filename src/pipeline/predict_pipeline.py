import sys
import pandas as pd
import os
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            #path of both preprocessor and model pkl files
            model_path = os.path.join("artifacts","model.pkl")
            preprocessor_path = os.path.join("artifacts","preprocessor.pkl")

            print("before loading")
            #loading model and preprocessor by pickle
            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)

            print("after loading pkl files")
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e,sys)
        
class CustomData:   #this class will convert the data into dataframe and return the dataframe
    def __init__(self,
                 gender:str,
                 race_ethnicity:str,
                 parental_level_of_education,
                 lunch:str,
                 test_preparation_course:str,
                 reading_score:int,
                 writing_score:int):        #this class take the inputs and self._______ are variables whose values are given input
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender":[self.gender],
                "race_ethnicity":[self.race_ethnicity],
                "parental_level_of_education":[self.parental_level_of_education],
                "lunch":[self.lunch],
                "test_preparation_course":[self.test_preparation_course],
                "reading_score":[self.reading_score],
              "writing_score":[self.writing_score]

            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e,sys)


