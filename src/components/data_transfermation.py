import pandas as pd
import numpy as np
import os
import sys
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import logging
from dataclasses import dataclass
from src.utilze import save_obj  # Ensure save_obj function is correctly implemented

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class ColumnTransformerConfig:
    # Path where the preprocessor object will be saved
    preprocess_obj_file_path: str = os.path.join('artifact', "preprocess_pkl")

class ColumnTransformerPipeline:
    def __init__(self):
        # Initialize the configuration for column transformation
        self.column_transformer_config = ColumnTransformerConfig()
        
    def get_column_transformer_config(self):
        """Constructs and returns a ColumnTransformer object for preprocessing."""
        try:
            # Define the feature lists for numeric and categorical features
            num_features = ['reading score','writing score']
            cat_features = ['gender','race/ethnicity','parental level of education','lunch','test preparation course']
            
            # Pipeline for numeric features: impute missing values and scale
            num_pipeline = Pipeline(
                steps=[
                    ("impute", SimpleImputer(strategy="median")),  # Impute missing values with median
                    ("standard", StandardScaler())  # Standardize features by removing the mean and scaling to unit variance
                ]
            )
            
            # Pipeline for categorical features: impute missing values, encode and scale
            cat_pipeline = Pipeline(
                steps=[
                    ("impute", SimpleImputer(strategy="most_frequent")),  # Impute missing values with most frequent value
                    ("encode", OneHotEncoder()),  # Encode categorical features as one-hot vectors
                    ("standard", StandardScaler(with_mean=False))  # Standardize features, but avoid mean centering due to encoding
                ]
            )
            
            logging.info(f"The categorical features are: {cat_features}")
            logging.info(f"The numerical features are: {num_features}")

            # Combine the numeric and categorical pipelines into a single ColumnTransformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, num_features),
                    ("cat_pipeline", cat_pipeline, cat_features)
                ]
            )
            return preprocessor
        
        except Exception as e:
            logging.error(f"Error in getting column transformer config: {e}")
            raise e

    def initiate_data_transformation(self, train_path: str, test_path: str):
        """Applies preprocessing steps to the training and test datasets."""
        try:
            # Load training and testing datasets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Successfully read the train and test datasets.")
            
            logging.info("Obtaining the preprocessor object.")
            preprocess_obj = self.get_column_transformer_config()
            
            target_column = 'math score'  # Column to be predicted

            # Processing training data
            X_train = train_df.drop(columns=[target_column])  # Features
            y_train = train_df[target_column]  # Target variable
            
            # Fit and transform the training data
            input_feature_train_arr = preprocess_obj.fit_transform(X_train)
            target_feature_train_arr = y_train.to_numpy()
            
            # Processing test data
            X_test = test_df.drop(columns=[target_column])  # Features
            y_test = test_df[target_column]  # Target variable
            
            # Transform the test data using the fitted preprocessor
            input_feature_test_arr = preprocess_obj.transform(X_test)
            target_feature_test_arr = y_test.to_numpy()
            
            train_arr = np.c_[input_feature_train_arr, target_feature_train_arr]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_arr]

            # Save the preprocessor object
            save_obj(
                file_path=self.column_transformer_config.preprocess_obj_file_path,
                obj=preprocess_obj
            )

            logging.info("Data transformation complete and preprocessor saved.")
            return train_arr, test_arr, preprocess_obj
        
        except Exception as e:
            logging.error(f"Error in data transformation: {e}")
            raise e
