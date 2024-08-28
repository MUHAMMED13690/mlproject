import pandas as pd
import numpy as np
import os
import logging
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transfermation import ColumnTransformerConfig
from src.components.data_transfermation import ColumnTransformerPipeline
from src.components.model_trainner import model_trainer_config
from src.components.model_trainner import model_trainer

# Configure logging to display messages at the INFO level and above,
# and format the log messages to include timestamp, log level, and message.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class DataIngestionConfig:
    # Define the paths for storing the training, testing, and raw datasets.
    train_data_path: str = os.path.join("artifact", 'train.csv')
    test_data_path: str = os.path.join("artifact", 'test.csv')
    raw_data_path: str = os.path.join("artifact", 'raw.csv')

class DataIngestion:
    def __init__(self):
        # Initialize the DataIngestionConfig object to access file paths.
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # Read the dataset from the specified CSV file path.
            df = pd.read_csv('notebook/data/StudentsPerformance.csv')
            logging.info('Dataset loaded as DataFrame')

            # Ensure the directory for saving the processed files exists.
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            
            # Save the raw dataset to the specified path.
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Saved raw data")

            # Split the dataset into training and testing sets.
            logging.info("Splitting the data into train and test sets")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            
            # Save the training and testing datasets to their respective paths.
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Data ingestion completed successfully")

            # Return the file paths of the training and testing datasets.
            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path
        except Exception as e:
            # Log the error message if an exception occurs and re-raise the exception.
            logging.error(f"An error occurred: {e}")
            raise e

if __name__ == "__main__":
    # Create an instance of the DataIngestion class and initiate data ingestion.
    ingestion = DataIngestion()
    train_data_path, test_data_path = ingestion.initiate_data_ingestion()

    # Create an instance of ColumnTransformerPipeline and apply data transformation
    column_transformation = ColumnTransformerPipeline()
    train_arr, test_arr, preprocess_obj= column_transformation.initiate_data_transformation(
        train_path=train_data_path,
        test_path=test_data_path
    )
    logging.info("Data transformation completed successfully")
    # Initialize and run the model trainer
    model_training = model_trainer()
    print(model_training.initiate_model_trainer(train_arr, test_arr))
