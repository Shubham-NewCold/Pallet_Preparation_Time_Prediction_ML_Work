import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor
from src.exception import CustomException
from src.utils import save_object, load_object
import logging
import sys
import os
from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self, train_df_final):
        try:
            logging.info("Split training and test input data")
            target_column_name = "target_variable"
        
            print("shape of train_df_final before splitting to X_train and y_train", train_df_final.shape)
        
            X_train = train_df_final.drop(columns=[target_column_name], axis=1)
            y_train = train_df_final[target_column_name]
        
            print("Checking for null values in the training data")
            print(X_train.isna().sum())
        
            X_train.to_csv("artifacts/X_train.csv", index=False)
        
            print("Shape of dataframes")
            print(X_train.shape)
            print(y_train.shape)
        
            lgbm_model = LGBMRegressor(random_state=42)
        
            param_grid = {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1],
                'max_depth': [-1, 10],
            }
        
            cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)
        
            grid_search = GridSearchCV(
                lgbm_model,
                param_grid,
                cv=cv_strategy,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
        
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
        
            model_scores = {
                'MSE': -cross_val_score(best_model, X_train, y_train, cv=cv_strategy, scoring='neg_mean_squared_error').mean(),
                'R2': cross_val_score(best_model, X_train, y_train, cv=cv_strategy, scoring='r2').mean(),
                'MAE': -cross_val_score(best_model, X_train, y_train, cv=cv_strategy, scoring='neg_mean_absolute_error').mean()
            }
        
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
        
            return model_scores, best_model
        
        except Exception as e:
            raise CustomException(e, sys)

    
    def predict_for_validation_set(self, test_df_final):
        try:
            target_column_name = "target_variable"
            X_test = test_df_final.drop(columns=[target_column_name], axis=1)
            y_test = test_df_final[target_column_name]
            
            model = load_object(self.model_trainer_config.trained_model_file_path)
            y_pred = model.predict(X_test)
            
            metrics = {
                'MSE': mean_squared_error(y_test, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                'MAE': mean_absolute_error(y_test, y_pred),
                'R2': r2_score(y_test, y_pred)
            }
            
            return metrics, y_pred
            
        except Exception as e:
            raise CustomException(e, sys)

        
        
    
    


            

            