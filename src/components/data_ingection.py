import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging

@dataclass

class data_ingection_cofig:
    train_data_path : str=os.path.join("artifict",'tarin.csv')
    test_data_path : str=os.path.join("artifict",'test.csv')
    rew_data_path : str=os.path.join("artifict",'rew.csv')

class data_ingection:
    def __init__(self):
        self.ingection_cofig= data_ingection_cofig()

    def intiate_data_ingection_cofig(self):
        logging.info("enter the the data ingection method or componet")
        try:
            df=pd.read_csv('StudentsPerformance.csv')
            logging.info('read the data set as data frame')

            os.makedirs(os.path.dirname(self.ingection_cofig.train_data_path),exist_ok=True)
            df.to_csv(self.ingection_cofig.rew_data_path,index=False,header=True)
            logging.info("train test split inserted")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.ingection_cofig.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingection_cofig.test_data_path,index=False,header=True)
            logging.info("the ingection of data is completed")
            return(
                self.ingection_cofig.test_data_path,
                self.ingection_cofig.train_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)
    
if __name__ == "__main__":
    ingestion = data_ingection()
    ingestion.intiate_data_ingection_cofig