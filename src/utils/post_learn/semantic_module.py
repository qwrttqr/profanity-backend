import os
import gc
import pandas as pd
import torch
from src.utils.text_analyzer import TextAnalyzer
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments,
                          Trainer)
from datasets import Dataset
from src.utils.load import semantic_directory_path
from src.utils.config import get_semantic_ver, save_semantic_ver
from db.utils.select_from_table import select_from_table
from db.utils.statemenents import select_from_model_answers_for_semantic, update_semantic_id
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from db.utils.update_table import get_id, update_table



class SemanticModule:
    SemanticModuleInstance = None

    def __new__(cls, *args, **kwargs):
        if cls.SemanticModuleInstance is None:
            cls.SemanticModuleInstance = super().__new__(cls)
        return cls.SemanticModuleInstance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.__model_ver = get_semantic_ver()
            self.__semantic_directory = semantic_directory_path / f'ver{self.__model_ver}'

            self.__tokenizer = AutoTokenizer.from_pretrained(self.__semantic_directory)
            self.__model = AutoModelForSequenceClassification.from_pretrained(
                self.__semantic_directory)

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
                item.update_semantic()
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
            model_lab = text_analyzer.get_semantic_labels(data[item['id']]['text'],
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

        df = pd.DataFrame(cleaned_data)
        df['labels'] = df['labels'].apply(lambda x: [float(i) for i in x])

        return df

    def post_learn(self, semantic_rows, text_analyzer: TextAnalyzer, threshold: float = 0.5):
        """
        Train a semantic classification model by given rows
        """
        dataframe = self.__prepare_data(semantic_rows=semantic_rows,
                                        text_analyzer=text_analyzer,
                                        threshold=threshold)

        rows = []
        for item in semantic_rows:
            rows.append({
                'id': item['id'],
                'phrase': item['text_after_processing']
            })

        dataset = Dataset.from_pandas(dataframe)
        dataset = dataset.shuffle(seed=42)

        train_test_split = dataset.train_test_split(test_size=0.1)
        train_dataset = train_test_split['train']
        eval_dataset = train_test_split['test']

        def tokenize_function(examples):
            return self.__tokenizer(examples["text"], padding="max_length", truncation=True,
                                    max_length=128)

        tokenized_train = train_dataset.map(tokenize_function, batched=True)
        tokenized_eval = eval_dataset.map(tokenize_function, batched=True)
        use_auc = True
        def compute_metrics(pred):
            logits, labels = pred
            probs = torch.sigmoid(torch.tensor(logits)).numpy()
            preds = (probs >= 0.5).astype(int)
            global use_auc

            try:
                auc = roc_auc_score(labels, probs, multi_class='ovr', average='macro')
            except:
                print('Cannot use auc_roc, using F1 instead')
                auc = None

            acc = accuracy_score(labels, preds)
            f1 = f1_score(labels, preds, average='macro')
            if auc is None:
                use_auc = False
                return {
                    'accuracy': acc,
                    'f1': f1
                }
            else:
                return {
                    'roc_auc': auc,
                    'accuracy': acc,
                    'f1': f1
                }

        if use_auc:
            training_args = TrainingArguments(
                eval_strategy="epoch",
                learning_rate=1e-5,
                per_device_train_batch_size=16,
                num_train_epochs=5,
                weight_decay=0.01,
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="roc_auc",
                report_to="none"
            )
        else:
            training_args = TrainingArguments(
                eval_strategy="epoch",
                learning_rate=1e-5,
                per_device_train_batch_size=16,
                num_train_epochs=5,
                weight_decay=0.01,
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="f1",
                greater_is_better=True,
                report_to="none"
            )

        trainer = Trainer(
            model=self.__model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.__model.to(device)

        if torch.cuda.is_available():
            self.__model.cuda()

        self.__model_ver += 1

        self.__semantic_directory = semantic_directory_path / f'ver{self.__model_ver}'
        os.makedirs(semantic_directory_path / f'ver{self.__model_ver}')

        gc.collect()
        torch.cuda.empty_cache()

        self.__model.save_pretrained(self.__semantic_directory)
        self.__tokenizer.save_pretrained(self.__semantic_directory)

        del self.__model
        del self.__tokenizer

        self.__model = AutoModelForSequenceClassification.from_pretrained(self.__semantic_directory)
        self.__tokenizer = AutoTokenizer.from_pretrained(self.__semantic_directory)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.__model.to(device)

        save_semantic_ver(self.__model_ver)

        self.__notify()

        for item in rows:
            labels = text_analyzer.get_semantic_labels(text=item['phrase'],
                                                       threshold=threshold)
            text_labels = {
                'toxic': labels[0],
                'insult': labels[1],
                'threat': labels[3],
                'dangerous': labels[4]
            }

            semantic_id = get_id(table_type='semantic_table',
                                 semantic_classes=text_labels)

            update_table(update_semantic_id,where ={'id': item['id']},
                         values={'semantic_id': semantic_id})


