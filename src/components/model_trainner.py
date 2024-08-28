from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
import os
import logging
from sklearn.metrics import r2_score
from src.utilze import save_obj, evaluate_model
from dataclasses import dataclass

@dataclass
class model_trainer_config:
    trained_model_file_path: str = os.path.join("artifact", 'model.pkl')

class model_trainer:
    def __init__(self):
        self.model_trainer_config = model_trainer_config()

    def initiate_model_trainer(self, train_arrey, test_arrey):
        try:
            logging.info("Splitting train and test data")
            x_train, y_train = train_arrey[:, :-1], train_arrey[:, -1]
            x_test, y_test = test_arrey[:, :-1], test_arrey[:, -1]

            # Debugging print statements
            print(f"x_train shape: {x_train.shape}")
            print(f"y_train shape: {y_train.shape}")
            print(f"x_test shape: {x_test.shape}")
            print(f"y_test shape: {y_test.shape}")

            models = {
                'LinearRegression': LinearRegression(),
                'Lasso': Lasso(),
                'Ridge': Ridge(),
                'RandomForestRegressor': RandomForestRegressor(),
                'AdaBoostRegressor': AdaBoostRegressor(),
                'DecisionTreeRegressor': DecisionTreeRegressor(),
                'KNeighborsRegressor': KNeighborsRegressor(),
                'XGBRegressor': XGBRegressor(),
                'Gradientboost': GradientBoostingRegressor()
            }

            model_report: dict = evaluate_model(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test, model=models)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]
            if best_model_score < 0.6:
                print("Best model not found")
            else:
                logging.info(f"Found the best model in data set: {best_model_name}")

            save_obj(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(x_test)
            r2 = r2_score(y_test, predicted)

            return r2
        except Exception as e:
            logging.error(f"An error occurred during model training: {e}")
            raise e
