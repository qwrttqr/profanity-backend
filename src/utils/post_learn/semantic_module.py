import datetime
import json
import os
import numpy as np
import pandas as pd
import torch
import gc
from typing import Any
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm.auto import tqdm, trange
from src.utils.text_analyzer import TextAnalyzer
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments,
                          Trainer, DataCollatorWithPadding)
from datasets import Dataset
from src.utils.file_work import FileManager
from db.utils.select_from_table import select_from_table
from db.utils.statemenents import select_from_model_answers_for_semantic, update_semantic_id
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, classification_report
from db.utils.update_table import get_id, update_table


class SemanticModule:
    SemanticModuleInstance = None

    def __new__(cls, *args, **kwargs):
        if cls.SemanticModuleInstance is None:
            cls.SemanticModuleInstance = super().__new__(cls)
        return cls.SemanticModuleInstance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.__file_manager = FileManager()
            self.__semantic_model_ver = self.__file_manager.get_semantic_ver()
            self.__observers = []
            self.initialized = True
            if os.path.exists(self.__file_manager.get_semantic_model_path(self.__semantic_model_ver)):

                self.__semantic_directory = self.__file_manager.get_semantic_model_path(self.__semantic_model_ver)

                self.__tokenizer = AutoTokenizer.from_pretrained(self.__semantic_directory)
                self.__model = AutoModelForSequenceClassification.from_pretrained(
                    self.__semantic_directory)
            else:
                print('Semantic model not found, starting up semantic model...')

                self.__startup_semantic_model()
                self.__semantic_directory = self.__file_manager.get_semantic_model_path(self.__semantic_model_ver)
                self.__tokenizer = AutoTokenizer.from_pretrained(self.__semantic_directory)
                self.__model = AutoModelForSequenceClassification.from_pretrained(
                    self.__semantic_directory)

                print('Semantic model started up')

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.__model.to(device)

            if torch.cuda.is_available():
                self.__model.cuda()

    def __notify(self):
        for item in self.__observers:
            try:
                item.update_semantic()
            except Exception as e:
                print('Error during observer notifying', e)
                raise Exception('Error during observer notifying')

    def __save_model(self, model_data: dict[str, Any]):
        """
            Saves model and its parts, updates semantic_directory variable

            Args:
                model_data: dict - dictionary with arbitrary model data
        """
        self.__file_manager.save_file(self.__semantic_directory, self.__model.save_pretrained)
        self.__file_manager.save_file(self.__semantic_directory, self.__tokenizer.save_pretrained)

        self.__file_manager.save_semantic_info(self.__semantic_model_ver, model_data)
        self.__semantic_directory = self.__file_manager.get_semantic_model_path(self.__semantic_model_ver)

    def __get_data_as_dataframe(self) -> pd.DataFrame:
        """
        Returns semantic rows as pandas DataFrame with columns [id, text_after_processing, toxic_class, insult_class, threat_class, dangerous_class]

        Returns:
            pd.DataFrame - DataFrame with data from database
        """
        print(select_from_table(statement=select_from_model_answers_for_semantic))
        return pd.DataFrame(select_from_table(statement=select_from_model_answers_for_semantic))

    # TODO recreate with actual 4 labels
    def __prepare_data(self, semantic_rows: list[dict[str, int | str | dict]],
                       text_analyzer,
                       threshold: float) -> pd.DataFrame:
        """
        Creates dataframe based on currently known data and new data.
        Args:
            semantic_rows: list[dict[str, int | str]] - array of new data
            text_analyzer: TextAnalyzer class that implement get_labels function
            threshold: float - threshold for labeling
        Returns:
            dataframe: pandas.DataFrame - dataframe with 2 columns(text & class)
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

    def __startup_semantic_model(self):
        """
        Startups semantic model and saves it to dedicated folder.
        """

        self.__semantic_model_ver = 1

        # Define labels to match your data format
        all_labels = ['toxic_class', 'insult_class', 'threat_class', 'dangerous_class']
        model_checkpoint = "cointegrated/rubert-tiny"

        # Initialize tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=len(all_labels))
        model.config.id2label = dict(enumerate(all_labels))
        model.config.label2id = {v: k for k, v in model.config.id2label.items()}
        if torch.cuda.is_available():
            model.cuda()

        # Data preparation function
        def prepare_dataset(df):
            """
            Prepare dataset from DataFrame with columns: text, toxic, insult, threat, dangerous
            """
            # Ensure all label columns exist and are binary (0 or 1)
            for label in all_labels:
                if label not in df.columns:
                    raise ValueError(f"Column '{label}' not found in DataFrame")
                df[label] = df[label].astype(int)

            # Create labels matrix
            labels_matrix = df[all_labels].values.astype(np.float32)

            # Create labels_mask (all ones since we don't have missing labels)
            labels_mask = np.ones_like(labels_matrix, dtype=np.float32)

            dataset_dict = {
                'text': df['text_after_processing'].tolist(),
                'labels': labels_matrix.tolist(),
                'labels_mask': labels_mask.tolist()
            }

            return Dataset.from_dict(dataset_dict)

        df = self.__get_data_as_dataframe()
        train_df, dev_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df[['toxic_class',
                                                                                             'insult_class',
                                                                                             'threat_class',
                                                                                             'dangerous_class']].sum(
            axis=1))

        # Prepare datasets
        train_dataset = prepare_dataset(train_df)
        dev_dataset = prepare_dataset(dev_df)

        # Tokenization
        def tokenize_function(examples):
            return tokenizer(examples["text"], truncation=True, max_length=512)

        train_tokenized = train_dataset.map(tokenize_function, batched=True)
        dev_tokenized = dev_dataset.map(tokenize_function, batched=True)

        # Data collator and dataloaders
        data_collator = DataCollatorWithPadding(tokenizer)
        batch_size = 64

        train_dataloader = DataLoader(
            train_tokenized.remove_columns('text'),
            batch_size=batch_size,
            drop_last=False,
            shuffle=True,
            num_workers=1,
            collate_fn=data_collator
        )

        dev_dataloader = DataLoader(
            dev_tokenized.remove_columns('text'),
            batch_size=batch_size,
            drop_last=False,
            shuffle=False,  # Changed to False for consistent evaluation
            num_workers=1,
            collate_fn=data_collator
        )

        # Evaluation function
        def evaluate_model(model, dev_dataloader):
            """
            Evaluate model and return AUC scores for each label
            """
            preds = []
            facts = []

            model.eval()
            for batch in tqdm(dev_dataloader, desc="Evaluating"):
                facts.append(batch['labels'].cpu().numpy())
                batch = {k: v.to(model.device) for k, v in batch.items()}

                with torch.no_grad():
                    outputs = model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        token_type_ids=batch.get('token_type_ids', None)
                    )
                preds.append(torch.sigmoid(outputs.logits).cpu().numpy())

            facts = np.concatenate(facts)
            preds = np.concatenate(preds)

            # Calculate AUC for each label
            results = []
            for i, label in enumerate(all_labels):
                try:
                    auc = roc_auc_score(facts[:, i], preds[:, i])
                    results.append(auc)
                except ValueError:
                    # Handle case where all labels are the same (no positive or negative examples)
                    results.append(0.0)

            return results, facts, preds

        # Memory cleanup function
        def cleanup():
            gc.collect()
            torch.cuda.empty_cache()

        # Training setup
        optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-5)
        gradient_accumulation_steps = 1
        window = 500
        cleanup_step = 100
        report_step = 1000  # Reduced for more frequent reporting
        loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')

        # Initial evaluation
        print("Initial evaluation:")
        eval_results, _, _ = evaluate_model(model, dev_dataloader)
        print(f"Initial AUC scores: {dict(zip(all_labels, eval_results))}")

        # Training loop
        num_epochs = 15
        ewm_loss = 0

        for epoch in trange(num_epochs, desc="Epochs"):
            model.train()
            tq = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")

            for i, batch in enumerate(tq):
                try:
                    batch = {k: v.to(model.device) for k, v in batch.items()}

                    outputs = model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        token_type_ids=batch.get('token_type_ids', None)
                    )

                    # Calculate loss for each class
                    loss_by_class = (loss_fn(outputs.logits, batch['labels']) * batch['labels_mask']).mean(dim=0)
                    loss = loss_by_class.sum()

                    loss.backward()

                except RuntimeError as e:
                    print(f'Error on epoch {epoch}, step {i}: {e}')
                    loss = torch.tensor(0.0, device=model.device)
                    cleanup()
                    continue

                # Gradient accumulation
                if (i + 1) % gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                # Periodic cleanup
                if i % cleanup_step == 0:
                    cleanup()

                # Update exponential moving average of loss
                w = 1 / min(i + 1, window)
                ewm_loss = ewm_loss * (1 - w) + loss.item() * w
                tq.set_description(f'Epoch {epoch + 1}, Loss: {ewm_loss:.4f}')

                # Periodic evaluation
                if i % report_step == 0 and i > 0:
                    eval_results, _, _ = evaluate_model(model, dev_dataloader)
                    model.train()
                    print(f'Epoch {epoch + 1}, Step {i}: Train Loss: {ewm_loss:.4f}')
                    print(f'Validation AUC: {dict(zip(all_labels, eval_results))}')

            # End of epoch evaluation
            eval_results, _, _ = evaluate_model(model, dev_dataloader)
            print(f'End of Epoch {epoch + 1}: Train Loss: {ewm_loss:.4f}')
            print(f'Validation AUC: {dict(zip(all_labels, eval_results))}')
            print("-" * 50)

        # Final evaluation
        print("\nFinal Model Evaluation:")
        eval_results, y_true, y_pred = evaluate_model(model, dev_dataloader)
        print(f'Final AUC scores: {dict(zip(all_labels, eval_results))}')

        # Save the model
        self.__model = model
        self.__tokenizer = tokenizer
        model_data = {
            'learning_date': str(datetime.date.today()),
            'model_ver': self.__semantic_model_ver
        }

        self.__save_model(model_data)

        self.__notify()

    @staticmethod
    def __transform_metrics(eval_results: dict[str, float],
                            report: dict[str, dict[str, float]],
                            filtered_labels: np.array) -> dict[str, dict[str, float]
                                                                    | dict[str, dict[str, float]]
                                                                    | dict[str, int]]:

        return {
            'training_metrics': eval_results,
            'classification_report': report,
            'class_distribution': {
                'toxic': int(filtered_labels[:, 0].sum()),
                'insult': int(filtered_labels[:, 1].sum()),
                'threat': int(filtered_labels[:, 2].sum()),
                'dangerous': int(filtered_labels[:, 3].sum())
            }
        }

    def get_model(self):

        return self.__model

    def get_tokenizer(self):

        return self.__tokenizer

    def attach(self, observer):
        if observer not in self.__observers:
            self.__observers.append(observer)

    def get_profanity_info(self) -> int:
        try:

            return self.__file_manager.load_file(
                (self.__file_manager.get_semantic_model_path(self.__semantic_model_ver) /
                 'model_info.json'),
                json.load,
                'r', False)

        except Exception as e:
            print(f'Error getting semantic ver: {str(e)}')
            raise Exception('Error during configuration semantic loading')

    def post_learn(self, semantic_rows,
                   text_analyzer: TextAnalyzer,
                   threshold: float = 0.5,
                   save_metrics: bool = False):
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
        dataset = dataset.shuffle()

        train_test_split = dataset.train_test_split(test_size=0.3)
        train_dataset = train_test_split['train']
        eval_dataset = train_test_split['test']

        def tokenize_function(examples):
            return self.__tokenizer(examples["text"], padding="max_length", truncation=True,
                                    max_length=128)

        tokenized_train = train_dataset.map(tokenize_function, batched=True)
        tokenized_eval = eval_dataset.map(tokenize_function, batched=True)

        def should_use_auc():
            for item in dataset:
                if all(label == 1 for label in item['labels']) or all(label == 0 for label in item[
                    'labels']):
                    return False

            return True

        def compute_metrics(pred):
            logits, labels = pred
            probs = torch.sigmoid(torch.tensor(logits)).numpy()
            preds = (probs >= 0.5).astype(int)
            use_auc = should_use_auc()

            if use_auc:
                auc = roc_auc_score(labels, probs, multi_class='ovr', average='macro')
            else:
                print('Cannot use auc_roc, using F1 instead')
                auc = None

            acc = accuracy_score(labels, preds)
            f1 = f1_score(labels, preds, average='macro')

            if auc is None:
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

        if should_use_auc():
            training_args = TrainingArguments(
                eval_strategy="epoch",
                learning_rate=1e-5,
                per_device_train_batch_size=16,
                num_train_epochs=5,
                weight_decay=0.01,
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="eval_roc_auc",
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
                metric_for_best_model="eval_f1",
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
        eval_results = trainer.evaluate()
        predictions = trainer.predict(tokenized_eval)

        # Process predictions
        probs = torch.sigmoid(torch.tensor(predictions.predictions)).numpy()
        preds = (probs >= threshold).astype(int)
        labels = predictions.label_ids

        # Select only the 4 relevant classes
        class_indices = [0, 1, 3, 4]  # toxic, insult, threat, dangerous
        filtered_labels = labels[:, class_indices]
        filtered_preds = preds[:, class_indices]

        # Generate metrics
        report = classification_report(
            filtered_labels,
            filtered_preds,
            target_names=['toxic', 'insult', 'threat', 'dangerous'],
            output_dict=True
        )

        metrics = SemanticModule.__transform_metrics(eval_results, report, filtered_labels)

        if save_metrics:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.__model.to(device)

            if torch.cuda.is_available():
                self.__model.cuda()

            self.__semantic_model_ver += 1

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
            model_data = {
                'learning_date': str(datetime.date.today()),
                'model_ver': self.__semantic_model_ver
            }
            self.__save_model(model_data)

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

                update_table(update_semantic_id, where={'id': item['id']},
                             values={'semantic_id': semantic_id})

        return metrics
