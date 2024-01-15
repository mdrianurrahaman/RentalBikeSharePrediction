
import os
import sys
from src.RentalBikeSharePrediction.exception import customexception
from src.RentalBikeSharePrediction.logger import logging
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df=pd.read_csv('notebooks/data/hour.csv')
            logging.info('Read the dataset as dataframe')

            # We will change the following Int column into a categorical column.
            cols = ['season','yr','mnth','hr','holiday','weekday','workingday','weathersit']

            for col in cols:
                df[col] = df[col].astype('category')

            # #The column 'instant' is very insignificant. Hence dropping that column
            # Similarly for column dteday too hence dropping it.
            # Since we have casual+registered=cnt and inferences are built from casual and registered records, 
            # let's drop them from dataframe df since these columns seem irrelevant for the model. 
            # Also from EDA, it is a given that increasing casual or registered users both will be profitable factor for the business.
            # Also temp and atemp are very highly corelated and their respective colinearities with cnt are also same. 
            # Hence dropping atemp since feeling temperature can be relatively less accurate compared to temperature.
            df=df.drop(['dteday','instant','casual', 'registered','atemp','holiday'],axis=1)


            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train test split initiated")

            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Ingestion of the data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )
        except Exception as e:
           logging.info("exception during data ingestion")
           raise customexception(e,sys)
