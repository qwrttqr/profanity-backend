from .load import files
from .text_prepar import text_preparator
from .db_collector import collect_information
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


class TextAnalyzer:
    '''
    Analyzing text on swear words and calculates text toxicity.
    For swear analytics we finding stemming of word. For  toxicity analytics
    finding lemmas.

    Methods:
    predict_profanity(input_text, threshold=0.5):
        Predict whether the input text contains profanity based on a specified threshold. 

    predict_proba_profanity(input_text):
        Predict the probability that the input text contains profanity.

    analyze_toxicity(text, aggregate, return_proba, threshold)
        Calculate (if return_proba = False) class for given text(0 - non-toxic, 1-toxic). Otherwise (if return_proba = True) returns based on aggregate param(if aggregate = True returns probability of a text to be toxic, if aggregate = False vector of a probability aspects).
    '''

    def __init__(self):

        self.__text_preparator = text_preparator
        self.__vectorizer = files['vectorizer_model']
        self.__model_profanity = files['ML_model']
        model_checkpoint = 'cointegrated/rubert-tiny-toxicity'
        self.__tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        self.__model_toxicity = AutoModelForSequenceClassification.from_pretrained(
            model_checkpoint)

        if torch.cuda.is_available():
            self.__model_toxicity.cuda()

    def predict_profanity(self, input_text: str, threshold: float = 0.5) -> int:
        '''
        Predicts does text has profane words or not.

        Args:
            input_text: str - input text.
            threshold: float - by default 0.5. Value between 0 and 1.
            Based on that number text will be labeled as contain profanity.

        Returns:

        '''

        predictions = self.__get_predict(input_text)
        preds = [0 if prob <
                      threshold else 1 for prob in predictions]

        return any(pred for pred in preds)

    def predict_proba_profanity(self, input_text: str) -> dict:
        predictions = self.__get_predict(input_text)

        return predictions

    def analyze(self, text: str, threshold: float = 0.5) -> dict[
        str, dict | int]:
        text_before_processing = text
        text_after = ' '.join(
            self.__text_preparator.prepare_text(text,
                                                basing=False))
        text_after_processing = text_after
        text_labels = self.__analyze_toxicity(text_after, threshold)
        profanity_label = self.predict_profanity(text_after)

        labels = {
            'text_labels': text_labels,
            'profanity_label': profanity_label
        }
        print(labels)
        # collect_information(text_before_processing,
        #                     text_after_processing,
        #                     analyzer_classes=text_labels,
        #                     profanity_class=profanity_label)
        return labels

    def __analyze_toxicity(self, text: str, threshold: float) -> list:
        '''
        Calculate toxicity of a text.
        Returns dictinary of text aspects:
            [toxic, insult, threat, dangerous]
        Threshould should be from 0 to 1. This param controls border to mark.

        ATTENTION
            The model for predictions got from here https://huggingface.co/cointegrated/rubert-tiny-toxicity
        '''

        with torch.no_grad():
            inputs = self.__tokenizer(
                text, return_tensors='pt',
                truncation=True, padding=True).to(
                self.__model_toxicity.device
            )

            proba = torch.sigmoid(
                self.__model_toxicity(
                    **inputs
                ).logits).cpu().numpy()

        if isinstance(text, str):
            proba = proba[0]

        probas = [(1 - proba[0]) * proba[-1]] + list(proba)[1:]
        probas = [int(i >= threshold) for i in probas]

        text_labels = {
            'toxic': probas[0],
            'insult': probas[1],
            'threat': probas[3],
            'dangerous': probas[4]
        }

        return text_labels

    def __get_predict(self, input_text) -> list[int]:
        input_text = self.__text_preparator.prepare_text(
            input_text, word_basing_method='stemming')

        vec_text = self.__vectorizer.transform(input_text)
        probabilities = self.__model_profanity.predict_proba(vec_text)[:, 1]

        return probabilities


text_analyzer = TextAnalyzer()
