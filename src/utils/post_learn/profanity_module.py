import datetime

import joblib
import pandas as pd
import numpy as np
import os
from src.utils.load import load_file
from sklearn.model_selection import train_test_split, GridSearchCV
from db.utils.select_from_table import select_from_table
from db.utils.statemenents import select_from_model_answers_for_profanity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.calibration import CalibratedClassifierCV
from src.utils.text_prepar import TextPreparation
from src.utils.text_analyzer import TextAnalyzer
from db.utils.update_table import get_id, update_table
from db.utils.statemenents import update_profanity_id
from src.utils.load import profanity_path
from src.utils.config import get_profanity_ver, save_profanity_info


class ProfanityModule:
    ProfanityModuleInstance = None

    def __new__(cls, *args, **kwargs):
        if cls.ProfanityModuleInstance is None:
            cls.ProfanityModuleInstance = super().__new__(cls)
        return cls.ProfanityModuleInstance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.__profanity_model_ver = get_profanity_ver()

            self.__model_path = (profanity_path / f'ver{self.__profanity_model_ver}' /
                                 'model.joblib')

            self.__vectorizer_path = profanity_path / \
                                     f'ver{self.__profanity_model_ver}' / 'vectorizer.joblib'

            self.__model = load_file(self.__model_path,
                                     joblib.load, 'rb',
                                     True)
            self.__vectorizer = load_file(self.__vectorizer_path,
                                          joblib.load, 'rb',
                                          True)
            self.initialized = True
            self.__observers = []
            self.__text_prepar = TextPreparation()

    def get_model(self):

        return self.__model

    def get_vectorizer(self):

        return self.__vectorizer

    def attach(self, observer):
        if observer not in self.__observers:
            self.__observers.append(observer)

    def __notify(self):
        for item in self.__observers:
            try:
                item.update_profanity()
            except Exception as e:
                print('Error during observer notifying', e)
                raise Exception('Error during observer notifying')

    def __prepare_word(self, elem: str) -> list[str]:
        return self.__text_prepar.prepare_text(elem,
                                               word_basing_method='stemming')

    def __prepare_data(self, profanity_rows: list[dict[str, int | str | dict]]) -> pd.DataFrame:
        """
        Creates dataframe based on currently known data and new data.
        Args:
            profanity_rows: list[dict[str, int | str]] - array of new data
        Returns:
            dataframe: pandas.DataFrame - dataframe with 2 columns(text & class)
        """
        rows = select_from_table(statement=select_from_model_answers_for_profanity)
        data = {}
        # Last key in frame
        last_key = 0
        for item in rows:
            prepared_row = self.__prepare_word(item['text_after_processing'])
            data[item['id']] = {
                'text': ' '.join(prepared_row),
                'profanity_class': item['profanity_class']
            }
            last_key = max(item['id'], last_key)

        for row in profanity_rows:
            if row['profanity_class'] == 1:
                data[row['id']]['profanity_class'] = row['profanity_class']
                for word in row['meta']['words']:
                    last_key += 1
                    processed_word = self.__prepare_word(word)[0]
                    data[last_key] = {
                        'text': processed_word,
                        'profanity_class': row['profanity_class']
                    }
            else:
                data[row['id']]['profanity_class'] = row['profanity_class']

        cleaned_data = []
        for value in data.values():
            row = {
                'text': value['text'],
                'profanity_class': value['profanity_class']
            }
            cleaned_data.append(row)
        df = pd.DataFrame(cleaned_data)

        return df

    @staticmethod
    def __transform_metrics(metrics: dict) -> dict[str, dict[str, float|int]]:
        return {
            'Не матное': metrics['0'],
            'Матное': metrics['1'],
            'accuracy': metrics['accuracy']
        }

    def post_learn(self, profanity_rows, text_analyzer: TextAnalyzer, save_model = False) -> dict[str, dict[str,float]|float]:
        """
        Train a profanity classification model with calibrated probabilities.

        When we got rows with data we are building a dataset by following rules:
        If profanity changed from 0 to 1 -> add profane word into the dataset with 1 label
        Otherwise mark the text as not containing profane words

        Args:
            profanity_rows: list - new rows for which model will do re-predictions
            text_analyzer: TextAnalyzer - analyzer module instance
            save_model: bool - Boolean flag on which we decide do we save new model or not
        Returns:
            dict|None - dictionary with metrics or None
        """
        dataframe = self.__prepare_data(profanity_rows)
        rows = []
        for item in profanity_rows:
            text_to_process = item['text_after_processing']

            prepared_phrase = self.__prepare_word(text_to_process)

            rows.append({
                'id': item['id'],
                'phrase': ' '.join(prepared_phrase)
            })

        X, y = dataframe.text, dataframe.profanity_class
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        vectorizer = CountVectorizer()
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        param_grid = {
            'C': np.array([0.001, 0.01, 0.1, 1., 10., 100.]),
            'class_weight': [None, 'balanced']
        }

        scoring = {
            'f1': 'f1',
            'recall': 'recall',
            'precision': 'precision',
            'accuracy': 'accuracy',
            'balanced_accuracy': 'balanced_accuracy'
        }

        # Grid search with cross-validation
        grid_clf_cv = GridSearchCV(
            estimator=LinearSVC(dual=False),
            param_grid=param_grid,
            scoring=scoring,
            refit='f1',
            return_train_score=True,
            cv=7
        )

        grid_clf_cv.fit(X_train_vec, y_train)

        final_clf = CalibratedClassifierCV(
            estimator=LinearSVC(**grid_clf_cv.best_params_)
        )
        final_clf.fit(X_train_vec, y_train)

        y_pred = final_clf.predict(X_test_vec)
        # Save metrics to show on front-end
        test_metrics = ProfanityModule.__transform_metrics(classification_report(y_test, y_pred, output_dict=True))
        print(test_metrics)
        if save_model:
            self.__model = final_clf
            self.__vectorizer = vectorizer

            self.__profanity_model_ver += 1

            os.makedirs(profanity_path / f'ver{self.__profanity_model_ver}')

            self.__model_path = (profanity_path / f'ver{self.__profanity_model_ver}' /
                                 'model.joblib')

            self.__vectorizer_path = (profanity_path / f'ver{self.__profanity_model_ver}' /
                                      'vectorizer.joblib')

            joblib.dump(self.__model, self.__model_path)
            joblib.dump(self.__vectorizer, self.__vectorizer_path)

            model_data = {
                'learning_date': str(datetime.date.today()),
                'model_ver': self.__profanity_model_ver
            }

            save_profanity_info(self.__profanity_model_ver, model_data, profanity_path / f'ver{self.__profanity_model_ver}' / 'model_info.json')

            self.__notify()

            for item in rows:
                class_ = text_analyzer.predict_profanity(item['phrase'])
                profanity_id = get_id(table_type='profanity_table', profanity_class=class_)
                update_table(update_profanity_id, where={'id': item['id']},
                             values={'profanity_id': profanity_id})
            return

        return test_metrics