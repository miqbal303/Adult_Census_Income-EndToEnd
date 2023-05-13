import sys
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from dataclasses import dataclass
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info('Splitting independent and dependent variables from train and test arrays')
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            # Define the classification models
            models = {
                'Logistic Regression': LogisticRegression(),
                'Random Forest': RandomForestClassifier(),
                'Support Vector Machine': SVC(probability=True),
                'GradientBoosting Classifier': GradientBoostingClassifier(),
                'XGB Classifier': XGBClassifier()
            }

            # Apply dimensionality reduction
            pca = PCA(n_components=5)
            X_train_pca = pca.fit_transform(X_train)
            X_test_pca = pca.transform(X_test)

            # Calculate evaluation metrics for each model
            accuracy_scores = {}
            auc_scores = {}
            classification_reports = {}

            for model_name, model in models.items():
                if model_name == 'Logistic Regression':
                    model.fit(X_train_pca, y_train)
                    y_pred = model.predict(X_test_pca)
                    y_pred_proba = model.predict_proba(X_test_pca)[:, 1]
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)[:, 1]

                accuracy = accuracy_score(y_test, y_pred)
                accuracy_scores[model_name] = accuracy

                report = classification_report(y_test, y_pred)
                classification_reports[model_name] = report

                auc = roc_auc_score(y_test, y_pred_proba)
                auc_scores[model_name] = auc

            # Select the best model based on the evaluation metrics
            best_model_name = None
            best_model_accuracy = 0

            for model_name, accuracy in accuracy_scores.items():
                if accuracy > best_model_accuracy:
                    best_model_name = model_name
                    best_model_accuracy = accuracy

            best_model_auc = auc_scores[best_model_name]

            print(f'Best Model: {best_model_name.capitalize()}')
            print(f'Accuracy: {best_model_accuracy}')
            print(f'AUC-ROC Score: {best_model_auc}')

            logging.info(f'Model Name: {best_model_name.capitalize()}, Accuracy Score: {best_model_accuracy}')

            print('\n====================================================================================\n')
            print(f'Classification Report: {classification_reports[best_model_name]}')

            logging.info(f'Classification Report: {classification_reports[best_model_name]}')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=models[best_model_name]
            )


        except Exception as e:
            logging.info('Exception occurred during model training')
            raise CustomException(e, sys)