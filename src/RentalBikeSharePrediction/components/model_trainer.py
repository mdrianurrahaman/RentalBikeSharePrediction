import os
import sys
from dataclasses import dataclass
import numpy as np

from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.neighbors import KNeighborsRegressor
from src.RentalBikeSharePrediction.exception import customexception
from src.RentalBikeSharePrediction.logger import logging

from src.RentalBikeSharePrediction.utils.utils import save_object,evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBoost Regressor": XGBRegressor(),
                "Bagging Regressor": BaggingRegressor(),
                "KNeighbors Regressor":KNeighborsRegressor()

            }

            # I am only mentioning best parameters i got for each model (using Gridsearchcv) while testing on EDA Jupyter notebook present in notebook folder.
            # Remaining parameter values i tested on will be kept commented
            params={
               
 
              "Decision Tree": {
                    'criterion':'friedman_mse',
                    'splitter':'random',
                    'max_depth':30,
                    'min_samples_leaf':3,
                    'min_samples_split':12,
                    'max_leaf_nodes':None
                },

              

                "Random Forest":{
                    'criterion':'friedman_mse',                 
                    'max_features':None,
                    'n_estimators': 256
                },

              
                "Gradient Boosting":{
                    'learning_rate': 0.2,
                    'max_depth'    : 8
                },


                "Linear Regression":{},


                # "CatBoosting Regressor":{
                #     "iterations": [1000],
                #     "learning_rate": [1e-3, 0.1],
                #     "depth": [1, 10],
                #     "subsample": [0.05, 1.0],
                #     "colsample_bylevel": [0.05, 1.0],
                #     "min_data_in_leaf": [1, 100]
                # },

                "XGBoost Regressor":{
                    "iterations": 1000,
                    "learning_rate":0.1,
                    "depth": 10,
                    "subsample":1.0,
                    "colsample_bylevel": 1.0,
                    "min_data_in_leaf":1
                },

                "Bagging Regressor":{
                    'base_estimator':None,
                    'n_estimators': 100,
                    'max_samples': 1.0,
                    'max_features': 1.0,
                    'bootstrap':True,
                    'bootstrap_features': False},

            
                "KNeighbors Regressor":{
                    'n_neighbors': 6,
                    'weights': 'distance'
                }
                
            }

            logging.info("Model evaluation stage and Hyperparameter tuning")

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,param=params)
                         

            # model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)
            print(model_report)
            print('\n====================================================================================\n')

            logging.info(f"Model Evaluation Report: {model_report}") 
            ## To get best model score from dict

            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            
            print(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            # putting a threshold that if best model score is not more than 60 percent then raise exception
            if best_model_score<0.6:
                raise customexception("No best model found")
            logging.info(f" print(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model 
            )

            predicted=best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return 'Best Model is ',best_model,' and its score is ',r2_square


        except Exception as e:
            raise customexception(e,sys)