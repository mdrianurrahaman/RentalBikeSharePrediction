from src.RentalBikeSharePrediction.components.data_ingestion import DataIngestion

import sys 
from src.RentalBikeSharePrediction.logger import logging 
from src.RentalBikeSharePrediction.exception  import customexception

obj = DataIngestion()
obj.initiate_data_ingestion()