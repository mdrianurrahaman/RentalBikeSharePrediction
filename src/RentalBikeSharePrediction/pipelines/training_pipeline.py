from src.RentalBikeSharePrediction.components.data_ingestion import DataIngestion

from src.RentalBikeSharePrediction.components.data_transformation import DataTransformation

from src.RentalBikeSharePrediction.components.model_trainer import ModelTrainer

import sys 
from src.RentalBikeSharePrediction.logger import logging 
from src.RentalBikeSharePrediction.exception  import customexception

obj = DataIngestion()
train_data_path,test_data_path =obj.initiate_data_ingestion()

data_transformation = DataTransformation()

train_arr,test_arr=data_transformation.initiate_data_transformation(train_data_path,test_data_path)

model_trainer_obj = ModelTrainer()
model_trainer_obj.initiate_model_trainer(train_arr,test_arr)

