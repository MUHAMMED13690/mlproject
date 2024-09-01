import pandas as pd
import numpy as np
import os
from src.utilze import load_object

class predict_pipline:
    def __init__(self):
        pass

    def predict(self,feature):
        try:
            model_path=os.path.join('artifact\model.pkl')
            preprocessor_path=os.path.join('artifact\preprocess_pkl')       
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(feature)
            preds=model.predict(data_scaled)
            return preds

        except Exception as e:
            raise e
        

class customdata:
    def __init__(  self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int):
        self.gender = gender

        self.race_ethnicity = race_ethnicity

        self.parental_level_of_education = parental_level_of_education

        self.lunch = lunch

        self.test_preparation_course = test_preparation_course

        self.reading_score = reading_score

        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        try:
            custom_data_input={
                'gender':[self.gender],
                'parental_level_of_education':[self.parental_level_of_education],
                'race_ethnicity':[self.race_ethnicity],
                'lunch':[self.lunch],
                'test_preparation_course':[self.test_preparation_course],
                'reading_score':[self.reading_score],
                'writing_score':[self.writing_score]
            }
            return pd.DataFrame(custom_data_input)
        except Exception as e:
            raise e
            
        
