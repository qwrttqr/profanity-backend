import pandas as pd
from src.utils.text_analyzer import TextAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.utils.load import semantic_directory_path
from db.utils.select_from_table import select_from_table
from db.utils.statemenents import select_from_model_answers_for_semantic
import torch


class SemanticModule:
    SemanticModuleInstance = None

    def __new__(cls, *args, **kwargs):
        if cls.SemanticModuleInstance is None:
            cls.SemanticModuleInstance = super().__new__(cls)
        return cls.SemanticModuleInstance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.__semantic_directory = semantic_directory_path
            self.__tokenizer = AutoTokenizer.from_pretrained(self.__semantic_directory)
            self.__model = AutoModelForSequenceClassification.from_pretrained(self.__semantic_directory)


            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.__model.to(device)

            if torch.cuda.is_available():
                self.__model.cuda()

            self.__observers = []

            self.initialized = True

    def get_model(self):

        return self.__model

    def get_tokenizer(self):

        return self.__tokenizer

    def attach(self, observer):
        if observer not in self.__observers:
            self.__observers.append(observer)

    def __notify(self):
        for item in self.__observers:
            try:
                item.update()
            except Exception as e:
                print('Error during observer notifying', e)
                raise Exception('Error during observer notifying')

    def __prepare_data(self, semantic_rows: list[dict[str, int | str | dict]],
                       text_analyzer,
                       threshold: float):
        """
        Creates dataframe based on currently known data and new data
        Args:
            semantic_rows: list[dict[str, int | str]] - array of new data
            text_analyzer: TextAnalyzer class that implement get_labels function
            threshold: float - threshold for labeling
        Returns:
            dataframe: pandas.DataFrame - dataframe with 2 columns

        """
        rows = select_from_table(statement=select_from_model_answers_for_semantic)
        data = {}
        for item in rows:
            model_lab = text_analyzer.get_semantic_labels(item['text_after_processing'],
                                                          threshold=threshold)

            label = [1 - item['toxic_class'],
                     item['insult_class'],
                     model_lab[2],
                     item['threat_class'],
                     item['dangerous_class']]

            data[item['id']] = {
                'text': item['text_after_processing'],
                'labels': label
            }

        for item in semantic_rows:
            model_lab = text_analyzer.get_semantic_labels(data[item['id']]['text_after_processing'],
                                                          threshold=threshold)
            label = [1 - item['toxic_class'],
                     item['insult_class'],
                     model_lab[2],
                     item['threat_class'],
                     item['dangerous_class']]

            data[item['id']]['labels'] = label

        cleaned_data = []
        for value in data.values():
            row = {
                'text': value['text'],
                'labels': value['labels']
            }
            cleaned_data.append(row)
        return pd.DataFrame(cleaned_data)

    def post_learn(self, semantic_rows, text_analyzer: TextAnalyzer, threshold: float = 0.5):
        """
        Train a profanity classification model with calibrated probabilities

        Returns:

        """
        dataframe = self.__prepare_data(semantic_rows = semantic_rows,
                                        text_analyzer=text_analyzer,
                                        threshold=threshold)

        print(dataframe)