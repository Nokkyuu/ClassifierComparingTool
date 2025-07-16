import logging
import warnings
from abc import ABC, abstractmethod
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator
from sklearn.exceptions import ConvergenceWarning
# import joblib

class MLHandler(ABC):
    def __init__(self, logger:logging.Logger = logging.getLogger()):
        self.model = self._initialize_model()
        self.params = {}
        if logger:
            self.logger = logger
        self.logger.info(f"Initialized {self.__class__.__name__} with model: {self.model}\n Parameters: {self.params}")

    @abstractmethod
    def _initialize_model(self) -> BaseEstimator:
        pass

    def grid_search(self, features, target, scoring='accuracy'):
        """performs grid search on the model with the given parameters.

        Args:
            features (DataFrame): _description_
            target (Series): _description_
            scoring (str, optional): Scoring goal for evaluation. Defaults to 'accuracy', other options are 'f1', 'recall', etc.

        Returns:
            model (Estimator) : best estimator found by grid search
            score (float): best score found by grid search
        """
        self.logger.info(f"Starting grid search with parameters: {self.params}")
        warnings.filterwarnings("ignore", category=ConvergenceWarning) #to avoid polluting output with convergence warnings
        grid_search = GridSearchCV(self.model, self.params, cv=5, scoring=scoring)
        grid_search.fit(features, target)
        self.logger.info(f"Best parameters found: {grid_search.best_params_}")
        return grid_search.best_estimator_, grid_search.best_score_

class DecisionTreeHandler(MLHandler):
    """Handler for Logistic Regression model. Instantiate with a logger.
    parameter for grid search are:
    - criterion: list of str, default ['gini', 'entropy']
    - max_depth: list of int, default [None, 10, 20, 30, 40]
    - min_samples_split: list of int, default [2, 5, 10]
    - min_samples_leaf: list of int, default [1, 2, 4]
    - max_features: list of str, default ['sqrt', 'log2']
    """
    def __init__(self, logger:logging.Logger = logging.getLogger(), params = None):
        super().__init__(logger)
        if params:
            self.params = params
        else:
            self.params = {
                'criterion': ['gini', 'entropy'],
                'max_depth': [None, 10, 20, 30, 40],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2']
            }

    def _initialize_model(self):
        return tree.DecisionTreeClassifier()

class RandomForestHandler(MLHandler):
    """Handler for Logistic Regression model. Instantiate with a logger.
    parameter for grid search are:
    - n_estimators: list of int, default [10, 50, 100, 200]
    - max_depth: list of int, default [None, 10, 20, 30, 40]
    - min_samples_split: list of int, default [2, 5, 10]
    - min_samples_leaf: list of int, default [1, 2, 4]
    - max_features: list of str, default ['sqrt', 'log2']
    """
    def __init__(self, logger:logging.Logger = logging.getLogger(), params = None):
        super().__init__(logger)
        if params:
            self.params = params
        else:
            self.params = {
                'n_estimators': [10, 20, 50],
                'max_depth': [None, 10, 20, 30, 40],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2']
            }
    def _initialize_model(self):
        return RandomForestClassifier()

class KNNHandler(MLHandler):
    """Handler for Logistic Regression model. Instantiate with a logger.
    parameter for grid search are:
    - n_neighbors: list of int, default [3, 5, 7, 9]
    - algorithm: list of str, default ['auto', 'ball_tree', 'kd_tree', 'brute']
    - leaf_size: list of int, default [30, 40, 50]
    """
    def __init__(self, logger:logging.Logger = logging.getLogger(), params = None):
        super().__init__(logger)
        if params:
            self.params = params
        else:
            self.params = {
                'n_neighbors': [3, 5, 7, 9],
                'weights' : ['uniform', 'distance'],
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                'leaf_size': [30, 40, 50]
            }
    def _initialize_model(self):
        return KNeighborsClassifier()

class LogisticRegressionHandler(MLHandler):
    """Handler for Logistic Regression model. Instantiate with a logger.
    parameter for grid search are:
    - C: list of float, default [0.001, 0.01, 0.1, 1, 10, 100]
    - penalty: list of str, default ['l2', 'l1', 'elasticnet', 'none']
    - solver: list of str, default ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    - max_iter: list of int, default [100, 200, 300]
    """
    def __init__(self, logger:logging.Logger = logging.getLogger(), params = None):
        super().__init__(logger)
        if params:
            self.params = params
        else:
            self.params = [
                #list of params because some combinations are not valid
                {'solver': ['saga'], 'penalty': ['l1', 'l2', 'elasticnet'], 'C': [0.001, 0.01, 0.1, 1, 10, 100], 'max_iter': [100, 200, 300]},
                {'solver': ['lbfgs', 'newton-cg', 'sag'], 'penalty': ['l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100], 'max_iter': [100, 200, 300]}
            ]
    def _initialize_model(self):
        return LogisticRegression()


if __name__ == "__main__":
    import pandas as pd
    from logging.config import fileConfig
    fileConfig("logging.ini")
    
    logger = logging.getLogger("debug")
    logger.info("Starting ML Handler example")

    df = pd.read_csv("data/iris.csv")
    features = df.drop(columns=["target"])
    logger .info(f"Features: {features.columns.tolist()}")

    target = df["target"]
    logger.info(f"Target: {target.name}")

    logreg_handler = LogisticRegressionHandler(logger)
    tree_handler = DecisionTreeHandler(logger)
    rf_handler = RandomForestHandler(logger)
    knn_handler = KNNHandler(logger)

    best_model, best_score = logreg_handler.grid_search(features, target, scoring='accuracy')
    logger.info(f"logistic regression: Best model: {best_model} \n Best score: {best_score}")
    best_model, best_score = tree_handler.grid_search(features, target, scoring='accuracy')
    logger.info(f"decision tree: Best model: {best_model} \n Best score: {best_score}")
    best_model, best_score = rf_handler.grid_search(features, target, scoring='accuracy')
    logger.info(f"random forest: Best model: {best_model} \n Best score: {best_score}")
    best_model, best_score = knn_handler.grid_search(features, target, scoring='accuracy')
    logger.info(f"knn: Best model: {best_model} \n Best score: {best_score}")
    logger.info("Finished ML Handler example")


