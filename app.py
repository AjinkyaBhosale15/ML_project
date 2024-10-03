from src.ml_project.logger import logging
from src.ml_project.exception import CustomException
from src.ml_project.components.data_ingestion import DataIngestion
from src.ml_project.components.data_ingestion import DataIngestionConfig
from src.ml_project.components.data_transformation import DataTransformationConfig, DataTransformation
from src.ml_project.components.model_trainer import ModelTrainerConfig, ModelTrainer
import sys

# Import dagshub and mlflow for logging
import dagshub
import mlflow

# Initialize DagsHub repository
dagshub.init(repo_owner='AjinkyaBhosale15', repo_name='ML_project', mlflow=True)

if __name__ == "__main__":
    logging.info("The execution has started")

    # Start an MLflow run to log parameters and metrics
    with mlflow.start_run():
        try:
            # Data ingestion
            data_ingestion = DataIngestion()
            train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

            # Log the paths as MLflow parameters
            mlflow.log_param('train_data_path', train_data_path)
            mlflow.log_param('test_data_path', test_data_path)

            # Data transformation
            data_transformation = DataTransformation()
            train_arr, test_arr, _ = data_transformation.initiate_data_transormation(train_data_path, test_data_path)

            # Log the shape of transformed datasets as parameters
            mlflow.log_param('train_data_shape', train_arr.shape)
            mlflow.log_param('test_data_shape', test_arr.shape)

            # Model training
            model_trainer = ModelTrainer()
            model_score = model_trainer.initiate_model_trainer(train_arr, test_arr)
            print(model_score)

            # Log the model score as a metric
            mlflow.log_metric('model_score', model_score)

        except Exception as e:
            logging.info("Custom Exception")
            raise CustomException(e, sys)
