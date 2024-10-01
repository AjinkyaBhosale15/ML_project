import os
import sys
import warnings
import pandas as pd
from src.ml_project.exception import CustomException
from src.ml_project.logger import logging
from src.ml_project.utils import read_sql_data
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import mysql.connector

# Suppress the warning from pandas about SQLAlchemy
warnings.filterwarnings("ignore", message="pandas only support SQLAlchemy connectable")

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            # Connect to MySQL database using mysql-connector
            conn = mysql.connector.connect(
                host='localhost',
                user='root',
                password='Tony@2020',
                database='college'
            )

            query = "SELECT * FROM student"  # Replace with your actual table name
            logging.info("Reading data from MySQL database")

            # Reading the data using mysql-connector and pandas
            df = pd.read_sql(query, conn)
            logging.info("Reading completed from MySQL database")

            # Ensure the directory for the data files exists
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save the raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            # Split data into train and test sets
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save train and test sets as CSV files
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data Ingestion completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
        logging.info(f"Train data saved at: {train_data_path}")
        logging.info(f"Test data saved at: {test_data_path}")
    except Exception as e:
        raise CustomException(e, sys)
