import sys
import os
import json
import joblib
import mlflow
from pathlib import Path
from mlflow.models import infer_signature
import numpy as np
from sklearn.linear_model import LinearRegression

from training.exception import ModelTrainingError, handle_exception
from training.custom_logging import info_logger, error_logger

from training.entity.config_entity import ModelTrainerConfig
from training.configuration_manager.configuration import ConfigurationManager

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config


    @staticmethod
    def filter_hyperparams(params):
        # Extract only the parameters related to the classifier (RandomForestClassifier)
        hyperparams = {key.replace('regressor__', ''): value for key, value in params.items() if key.startswith('regressor__')}
        return hyperparams

    def load_transformed_data(self):
        try:
            info_logger.info("Loading Final Training Transformed Data")

            final_train_data_path = os.path.join(self.config.final_train_data_path, 'Train.npz')
            final_test_data_path = os.path.join(self.config.final_test_data_path,"Test.npz")

            final_train_data = np.load(final_train_data_path, allow_pickle=True)
            final_test_data = np.load(final_test_data_path, allow_pickle=True)

            xtrain = final_train_data["xtrain"]
            xtest = final_test_data["xtest"]
            ytrain = final_train_data["ytrain"]
            ytest = final_test_data["ytest"]


            info_logger.info("Loaded Final Training Transformed Data")

            return xtrain, xtest, ytrain, ytest
        except Exception as e:
            handle_exception(e, ModelTrainingError)

    def train_model(self, xtrain,xtest, ytrain, y_test):
        try:
            mlflow.set_tracking_uri("http://127.0.0.1:5000")
            mlflow.set_experiment("Final Model Training Experiment")
            with mlflow.start_run(run_name="Final Model Training"):
                
                # Step 1: Load the best estimator from MLflow
                best_model_uri = "models:/BestEstimatorModel/latest"  # Retrieve latest registered model
                best_model = mlflow.sklearn.load_model(best_model_uri)
                
                # Step 2: Extract hyperparameters from the best model
                best_hyperparams = self.filter_hyperparams(best_model.get_params())
                
                mlflow.log_params(best_hyperparams)

                # Step 3: Train the final model using the best hyperparameters
                final_model = LinearRegression(**best_hyperparams)
                final_model.fit(xtrain, ytrain)

                # Step 4: Evaluate the final model on the test set
                final_test_score = final_model.score(xtest, y_test)
                mlflow.log_metric("R2 final test set", final_test_score)

                signature = infer_signature(xtrain, final_model.predict(xtrain))
                mlflow.sklearn.log_model(final_model, artifact_path="final_model", signature=infer_signature(xtrain, final_model.predict(xtrain)), registered_model_name="FinalModel")

            return final_model
        except Exception as e:
            handle_exception(e, ModelTrainingError)

    def save_model(self, model):
        try:
            info_logger.info("Saving final model started")

            model_path = os.path.join(self.config.root_dir, "final_model.joblib")
            joblib.dump(model, model_path)


            with open(self.config.STATUS_FILE, "w") as f:
                f.write(f"Model Training status: True")

            info_logger.info("Final model saved")
        except Exception as e:
            handle_exception(e, ModelTrainingError)

if __name__ == "__main__":
    config = ConfigurationManager()
    model_trainer_config = config.get_model_trainer_config()

    model_trainer = ModelTrainer(config = model_trainer_config)
    xtrain, xtest, ytrain, ytest = model_trainer.load_transformed_data()
    model = model_trainer.train_model(xtrain, xtest, ytrain, ytest)
    model_trainer.save_model(model)