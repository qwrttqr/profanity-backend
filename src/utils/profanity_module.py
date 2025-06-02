import joblib
import pandas as pd
import numpy as np
from .load import load_file
from sklearn.model_selection import train_test_split
from db.utils.select_from_db import select_from_table
from db.utils.statemenents import select_from_model_answers_for_profanity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.calibration import CalibratedClassifierCV


class ProfanityModule:
    ProfanityModuleInstance = None
    def __new__(cls, *args, **kwargs):
        if cls.ProfanityModuleInstance is None:
            cls.ProfanityModuleInstance = super().__new__(cls)
        return cls.ProfanityModuleInstance

    def __init__(self, model_path, vectorizer_path):
        if not hasattr(self, 'initialized'):

            self.__model = load_file(model_path,
                                     joblib.load, 'rb',
                                     True)
            self.__vectorizer = load_file(vectorizer_path,
                                          joblib.load, 'rb'
                                          , True)
            self.initialized = True
            self.__model_path = model_path
            self.__vectorizer_path = vectorizer_path
            self.__observers = []

    def get_model(self):

        return self.__model

    def get_vectorizer(self):

        return self.__vectorizer

    def attach(self, observer):
        if observer not in self.__observers:
            self.__observers.append(observer)

    def notify(self):
        for item in self.__observers:
            try:
                item.update()
            except Exception as e:
                print('Error during observer notifying', e)
    def __prepare_data(self, profanity_rows: list[dict[str, int | str | dict]]):
        '''
        Creates dataframe based on currently known data and new data
        Args:
            profanity_rows: list[dict[str, int | str]] - array of new data

        Returns:
            dataframe: pandas.DataFrame - dataframe with 2 columns?

        '''
        rows = select_from_table(statement=select_from_model_answers_for_profanity)
        data = {
            'text': [],
            'profanity_class': []
        }

        for item in rows:
            data['text'].append(item['text_after_processing'])
            data['profanity_class'].append(item['profanity_class'])

        for item in profanity_rows:
            for elem in item['meta']['profane_words']:
                data['text'].append(elem)
                data['profanity_class'].append(item['profanity_class'])

        return pd.DataFrame(data)

    def post_learn(self, profanity_rows):
        """
        Train a profanity classification model with calibrated probabilities

        Returns:
            tuple: (trained_model, vectorizer, test_metrics)
        """
        dataframe = self.__prepare_data(profanity_rows)

        X, y = dataframe.text, dataframe.profanity_class
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Fit vectorizer only on training data to prevent data leakage
        vectorizer = CountVectorizer()
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        # Grid search parameters
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

        # Train final calibrated model with best parameters
        final_clf = CalibratedClassifierCV(
            estimator=LinearSVC(**grid_clf_cv.best_params_)
        )
        final_clf.fit(X_train_vec, y_train)

        # Evaluate on test set
        y_pred = final_clf.predict(X_test_vec)
        test_metrics = classification_report(y_test, y_pred, output_dict=True)

        self.__model = final_clf
        self.__vectorizer = vectorizer

        joblib.dump(self.__model, self.__model_path)
        joblib.dump(self.__vectorizer, self.__vectorizer_path)

        self.notify()
