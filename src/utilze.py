import pickle
import os
from sklearn.metrics import r2_score

# Function to save an object (e.g., trained model) to a file
def save_obj(file_path, obj):
    try:
        # Ensure the directory exists
        dir_name = os.path.dirname(file_path)
        os.makedirs(dir_name, exist_ok=True)
        
        # Save the object to the specified file path using pickle
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        print(f"An error occurred: {e}")

from sklearn.metrics import r2_score

# Function to evaluate multiple models
def evaluate_model(X_train, y_train, X_test, y_test, models):
    """
    Trains multiple models and evaluates them on test data without hyperparameter tuning.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target variable.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test target variable.
        models (dict): Dictionary of model names and instantiated model objects.

    Returns:
        dict: Dictionary containing model names and their test R2 scores.
    """
    try:
        report = {}  # Dictionary to store the evaluation results

        for model_name, model in models.items():
            print(f"Training model: {model_name}")

            # Train the model with default parameters
            model.fit(X_train, y_train)

            # Make predictions on training and test data
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Calculate R2 scores for training and test data
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            # Save the test score in the report
            report[model_name] = test_model_score

            print(f"{model_name}: Train R2 Score = {train_model_score:.4f}, Test R2 Score = {test_model_score:.4f}")

        return report  # Return the dictionary containing model names and their test R2 scores
    except Exception as e:
        raise e  # Raise any exceptions that occur during model evaluation
def load_object(file_path):
    try:
        # Open the file in read-binary mode
        with open(file_path, "rb") as file_obj:
            # Load the pickled object
            return pickle.load(file_obj)
    except Exception as e:
        raise e  # Raise any exceptions that occur during loading

