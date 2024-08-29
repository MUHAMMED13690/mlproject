import pickle
import os
from sklearn.metrics import r2_score

def save_obj(file_path, obj):
    try:
        dir_name = os.path.dirname(file_path)
        os.makedirs(dir_name, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        print(f"An error occurred: {e}")

from sklearn.metrics import r2_score

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
        report = {}

        for model_name, model in models.items():
            print(f"Training model: {model_name}")

            # Train the model with default parameters
            model.fit(X_train, y_train)

            # Evaluate the model on training and test data
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            # Save test score in report
            report[model_name] = test_model_score

            print(f"{model_name}: Train R2 Score = {train_model_score:.4f}, Test R2 Score = {test_model_score:.4f}")

        return report
    except Exception as e:
        raise e
