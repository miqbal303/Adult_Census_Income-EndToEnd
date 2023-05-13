import sys
from dataclasses import dataclass
import os

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object


@dataclass
class DataTransformationconfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationconfig()

    def get_data_transformation_object(self):
        try:
            logging.info('Data Transformation initiated')
            # Selecting numeric columns
            num_cols = ['Age', 'Final_Weight', 'Education_Number_of_Years', 'Capital_gain',
                        'Capital_loss', 'Hours_per_week']
            # Define the columns to be encoded using LabelEncoder
            cat_cols = ['Workclass', 'Marital_status', 'Occupation', 'Relationship',
                        'Race', 'Sex', 'Native_country']
            # Define the columns to be encoded using OrdinalEncoder
            ordinal_cols = ['Education']
            # Define the target column
            target_cols = ['Income']

            # Define the mapping for ordinal encoding
            education_mapping = {
                'Preschool': 0,
                '1st-4th': 0,
                '5th-6th': 1,
                '7th-8th': 1,
                '9th': 2,
                '10th': 2,
                '11th': 2,
                '12th': 2,
                'HS-grad': 3,
                'Some-college': 4,
                'Assoc-voc': 5,
                'Assoc-acdm': 5,
                'Bachelors': 6,
                'Masters': 7,
                'Prof-school': 7,
                'Doctorate': 7
            }
            logging.info('Pipeline Initiated')

            # Define the numerical pipeline
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('std', StandardScaler())
            ])

            # Define the categorical pipeline
            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])

            # Define the ordinal pipeline
            ord_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ord', OrdinalEncoder(categories=[list(education_mapping.keys())])),
                ('std', StandardScaler())
            ])

            # Define the preprocessor
            preprocessor = ColumnTransformer([
                ('num', num_pipeline, num_cols),
                ('cat', cat_pipeline, cat_cols),
                ('ord', ord_pipeline, ordinal_cols)
            ])

            logging.info('Pipeline Completed')

            return preprocessor

        except Exception as e:
            logging.info('Error in Data Transformation')
            raise CustomException(e, sys)



    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')

            logging.info('Obtaining preprocessing object')

            preprocessing_obj = self.get_data_transformation_object()

            income_map = {'<=50K': 0, '>50K': 1}

            train_df['Income'] = train_df['Income'].map(income_map)
            test_df['Income'] = test_df['Income'].map(income_map)
            logging.info('Target column encoded')

            train_df.replace('?', np.nan, inplace=True)
            test_df.replace('?', np.nan, inplace=True)
            logging.info('Replacing ? with NaN')

            logging.info(f'Train Dataframe head: \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe head: \n{test_df.head().to_string()}')

            target_column_name = 'Income'
            drop_columns = [target_column_name]

            input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=drop_columns, axis=1)
            target_feature_test_df = test_df[target_column_name]
            logging.info(f'Train Dataframe head: \n{input_feature_train_df.head().to_string()}')
            logging.info(f'Test Dataframe head: \n{target_feature_train_df.head().to_string()}')

            # Transforming using preprocessor obj
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on training and testing datasets.")

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]


            # Apply SMOTE for oversampling
            smote = SMOTE(random_state=42)
            train_arr_resampled, target_feature_train_resampled = smote.fit_resample(
                train_arr[:, :-1], train_arr[:, -1]
            )
            
            logging.info('SMOTE oversampling applied.')

            train_arr_resampled = np.c_[train_arr_resampled, target_feature_train_resampled]
            logging.info(f'Train Dataframe head: \n{train_arr_resampled.head().to_string()}')
            logging.info('Resampled train array created.')

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            logging.info('Preprocessor pickle file saved')

            return (
                train_arr_resampled,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            logging.info('Exception occurred in the initiate_data_transformation')
            raise CustomException(e, sys)


