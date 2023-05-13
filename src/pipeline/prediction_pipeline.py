import sys
import os
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model_path=os.path.join('artifacts','model.pkl')

            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)

            data_scaled=preprocessor.transform(features)

            pred=model.predict(data_scaled)
            return pred
            

        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e,sys)
        

class CustomData:
    def __init__(self,
                 Age:float,
                 Final_Weight:float,
                 Education_Number_of_Years:float,
                 Capital_gain:float,
                 Capital_loss:float,
                 Hours_per_week:float,
                 Workclass:str,
                 Education:str,
                 Marital_status:str,
                 Occupation:str,
                 Relationship:str,
                 Race:str,
                 Sex:str,
                 Native_country:str):

        
        self.Age=Age
        self.Final_Weight = Final_Weight
        self.Education_Number_of_Years=Education_Number_of_Years
        self.Capital_gain=Capital_gain
        self.Capital_loss= Capital_loss
        self.Hours_per_week=Hours_per_week
        self.Workclass=Workclass
        self.Education = Education
        self.Marital_status = Marital_status
        self.Occupation = Occupation
        self.Relationship = Relationship
        self.Race = Race
        self.Sex = Sex
        self.Native_country=Native_country 

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'Age':[self.Age],
                'Education_Number_of_Years':[self.Education_Number_of_Years],
                'Final_Weight':[self.Final_Weight],
                'Capital_gain':[self.Capital_gain],
                'Capital_loss':[self.Capital_loss],
                'Hours_per_week':[self.Hours_per_week],
                'Workclass':[self.Workclass],
                'Education':[self.Education],
                'Marital_status':[self.Marital_status],
                'Occupation':[self.Occupation],
                'Relationship':[self.Relationship],
                'Race':[self.Race],
                'Sex':[self.Sex],
                'Native_country':[self.Native_country]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e,sys)

