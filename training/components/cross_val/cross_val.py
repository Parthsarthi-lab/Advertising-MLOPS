import os
import sys
import json
import mlflow
import joblib
from pathlib import Path

import pandas as pd
import numpy as np

from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold, train_test_split

from training.custom_logging import info_logger, error_logger
from training.exception import CrossValError, handle_exception

from training.configuration_manager.configuration import ConfigurationManager
from training.entity.config_entity import CrossValConfig

class CrossVal:
    def __init__(self, config: CrossValConfig):
        self.config = config


    @staticmethod
    def is_json_serializable(value):
      """
      Check if a value is JSON serializable.
      """
      try:
          json.dumps(value)
          return True
      except (TypeError, OverflowError):
          return False
      
    def load_ingested_data(self):
        try:
            info_logger.info("Cross Validation Component started")
            info_logger.info("Loading ingested data")

            data_path = self.config.data_dir

            df = pd.read_csv(data_path, index_col=0)
            df.reset_index(drop=True, inplace=True)

            X = df.drop("sales", axis=1)
            y = df["sales"]

            info_logger.info("Ingested data loaded")
            return X, y

        except Exception as e:
            handle_exception(e, CrossValError)


    def split_data_for_final_train(self, X, y):
        try:
            info_logger.info("Data split for final train started")
            
            xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)
            
            info_logger.info("Data split for final train completed")
            return xtrain, xtest, ytrain, ytest
        except Exception as e:
            handle_exception(e, CrossValError)
    
    def save_data_for_final_train(self, xtrain, xtest, ytrain, ytest):
        try:
            info_logger.info("Saving data for final train started")

            final_train_data_path = self.config.final_train_data_path
            final_test_data_path = self.config.final_test_data_path

            # Save xtrain and ytrain  to Train.npz
            # Save xtest and ytest to Test.npz
            np.savez(os.path.join(final_train_data_path, 'Train.npz'), xtrain=xtrain, ytrain=ytrain)
            np.savez(os.path.join(final_test_data_path, 'Test.npz'),  xtest=xtest, ytest=ytest)

            info_logger.info("Saved data for final train")
        except Exception as e:
            handle_exception(e, CrossValError)
    


    def run_cross_val(self, X, y):
        try:
            mlflow.set_experiment("Cross Validation Experiment")
            mlflow.set_tracking_uri("http://127.0.0.1:5000")
            # Step 1: Define the pipeline and parameter grid
            numeric_features = X.columns
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features),
                ]
            )

            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('regressor', LinearRegression())
            ])

            param_grid = {'regressor__fit_intercept': [True, False]}

            grid_search = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                scoring='r2',
                cv=5,
                verbose=2,
                return_train_score=True
            )

            # Step 2: Fit the GridSearchCV
            grid_search.fit(X, y)

            # Step 3: Log each hyperparameter combination as a separate MLflow run
            results = pd.DataFrame(grid_search.cv_results_)
            for idx, row in results.iterrows():
                with mlflow.start_run(nested=True):  # Create a nested run for each combination
                    # Log the parameters and metrics
                    mlflow.log_params(row["params"])
                    mlflow.log_metric("mean_train_score", row['mean_train_score'])
                    mlflow.log_metric("mean_test_score", row['mean_test_score'])

                    # Save and log the intermediate model
                    intermediate_model_path = os.path.join(self.config.best_model_params, f'model_combination_{idx}.joblib')
                    pipeline.set_params(**row['params'])  # Set parameters for this combination
                    pipeline.fit(X, y)  # Fit the pipeline
                    joblib.dump(pipeline, intermediate_model_path)

                    y_pred = pipeline.predict(X)
                    signature = infer_signature(X, y_pred)

                    mlflow.sklearn.log_model(pipeline,artifact_path="intermediate_models", signature= signature)

            # Step 4: Log the best estimator separately
            with mlflow.start_run(nested=True, run_name="Best Estimator"):  # Create a nested run for the best estimator
                best_params = grid_search.best_params_
                best_score = grid_search.best_score_

                mlflow.log_params(best_params)
                mlflow.log_metric("best_score", best_score)

                # Save and log the best estimator
                best_model_path = os.path.join(self.config.best_model_params, 'best_model.joblib')
                joblib.dump(grid_search.best_estimator_, best_model_path)

                y_pred = grid_search.best_estimator_.predict(X)
                signature = infer_signature(X, y_pred)

                # Save the best model parameters as a JSON file
                best_model_params_path = os.path.join(self.config.best_model_params, 'best_params.json')
                best_model_params = grid_search.best_estimator_.get_params()
                serializable_params = {k: v for k, v in best_model_params.items() if self.is_json_serializable(v)}

                with open(best_model_params_path, 'w') as f:
                    json.dump(serializable_params, f, indent=4)

                mlflow.sklearn.log_model(grid_search.best_estimator_, artifact_path="best_model", signature= signature, registered_model_name="BestEstimatorModel")

            info_logger.info("Cross Validation completed")
        except Exception as e:
            handle_exception(e, CrossValError)


if __name__ == "__main__":
    config = ConfigurationManager()
    cross_val_config = config.get_cross_val_config()

    cross_val = CrossVal(config=cross_val_config)

    # Load the features and target
    X,y = cross_val.load_ingested_data()

    # Split the data into train and test sets for final train
    xtrain, xtest, ytrain, ytest = cross_val.split_data_for_final_train(X,y)

    # Save xtrain, xtest, ytain, ytest to be used final train
    cross_val.save_data_for_final_train(xtrain, xtest, ytrain, ytest)

    # Run cross validation
    cross_val.run_cross_val(xtrain, ytrain)
