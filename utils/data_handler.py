import pandas as pd
import logging

class DataHandler:
    """Handler for loading and manipulating data. Instantiate with a logger."""
    def __init__(self, logger:logging.Logger = logging.getLogger()):
        self.data = None
        self.features = None
        self.target = None
        self.logger = logger

    def load_data(self, file, delimiter: str = ','): 
        """Load data from a CSV file.

        Args:
            path (str): path to the CSV file
            delimiter (str): delimiter used in the CSV file, default is ','

        Returns:
            list: list of column names in the data
        """
        try:
            self.data = pd.read_csv(file, delimiter=delimiter)
            self.logger.info(f"Data loaded from {file}")
            return self.data.columns
        except FileNotFoundError:
            self.logger.error(f"Data file not found at {file}")
            return None
        
    def feature_select(self, features: list):
        """create a new DataFrame with the selected features.

        Args:
            features (list): set of feature names to select
        """
        if self.data is not None:
            if not set(features).issubset(self.data.columns):
                self.logger.error("Some features are not in the data.")
                return
            else:
                self.features = self.data[features]
                self.logger.info(f"Selected features: {features}")
        else:
            self.logger.error("No data loaded to select features from.")
            return 
    
    def target_select(self, target: str):
        """create a new DataFrame with the target variable.

        Args:
            target (str): name of the target variable
            """
        if self.data is not None and target in self.data.columns:
            self.target = self.data[target]
            self.logger.info(f"Selected target: {target}")   
        else:
            self.logger.error(f"Target '{target}' not found in data.")

if __name__ == "__main__":
    # For testing purposes, you can run this script directly
    from logging.config import fileConfig
    fileConfig("logging.ini")
    logger = logging.getLogger("debug")
    
    data_handler = DataHandler(logger)
    columns = data_handler.load_data('data/iris.csv')
    if columns:
        logger.info(f"Columns in the data: {columns}")
        features = columns.to_list()[:-1]  #last column is target here
        data_handler.feature_select(features)
        data_handler.target_select('target')
            