import pandas as pd
import numpy as np
import os
import sys
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass

@dataclass

class column_transformer_config:
    preproces_obj_file_path=os.path.join('artifect',"preprocess_pkl")

class columtransfer:
    def __init__(self):
        self.column_transfer=column_transformer_config()
        
    def get_column_transfer_config(self):
        try:
                num_features=['reading score', 'writing score']
                cat_features=['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
                num_pipline=Pipeline(
                     steps=[
                          ("impute",SimpleImputer(strategy="median")),
                          ("standerd",StandardScaler())
                          ]
                          )
                cat_pipline=Pipeline(
                     steps=[
                          ("impute",SimpleImputer(strategy="most_frequent")),
                          ("encode",OneHotEncoder()),
                          ("standerd",StandardScaler())
                          ]
                          )
                logging.info(f"The catogrical features are:{cat_features}")
                logging.info(f"The numerical features are:{num_features}")

                preprocessor=ColumnTransformer(
                     ("num_pipline",num_pipline,num_features),
                     ("cat_pipline",cat_pipline,cat_features)
                )
                return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
