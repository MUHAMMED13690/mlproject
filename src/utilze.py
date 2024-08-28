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

def evaluate_model(x_train, x_test, y_train, y_test, model):
    """
    Evaluates multiple models and returns a dictionary with model names and their corresponding r2 scores.

    Parameters:
    - x_train: Training features
    - x_test: Testing features
    - y_train: Training labels
    - y_test: Testing labels
    - models: Dictionary of model names and instances

    Returns:
    - model_report: Dictionary of model names and their r2 scores
    """
    model_report = {}
    for model_name, model_instance in model.items():
        model_instance.fit(x_train, y_train)
        y_test_predict = model_instance.predict(x_test)

        # Flatten y_test_predict if y_test is 1-dimensional
        if y_test.ndim == 1:
            y_test_predict = y_test_predict.flatten()

        # Calculate r2 score
        test_model_score = r2_score(y_test, y_test_predict)
        model_report[model_name] = test_model_score

    return model_report
