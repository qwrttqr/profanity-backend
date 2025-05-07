import torch
from utils import model_load, vectorizer_load, TextPreparation
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from fastapi import FastAPI
from fastapi import Query
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
class TextAnalyzer:
    """
    Analyzing text on swear words and calculates text toxicity.
    For swear analytics we finding stemming of word. For  toxicity analytics finding lemmas

    Methods:
    predict_profanity(input_text, threshold=0.5):
        Predict whether the input text contains profanity based on a specified threshold. 

    predict_proba_profanity(input_text):
        Predict the probability that the input text contains profanity.
    
    analyze_toxicity(text, aggregate, return_proba, threshold)
        Calculate (if return_proba = False) class for given text(0 - non-toxic, 1-toxic). Otherwise (if return_proba = True) returns based on aggregate param(if aggregate = True returns probability of a text to be toxic, if aggregate = False vector of a probability aspects).
    """
    def __init__(self):

        self.__text_preparator = TextPreparation()
        self.__vectorizer = vectorizer_load()
        self.__model_profanity = model_load()
        model_checkpoint = 'cointegrated/rubert-tiny-toxicity'
        self.__tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        self.__model_toxicity = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)

        if torch.cuda.is_available():
            self.__model_toxicity.cuda()

    def predict_profanity(self, input_text: str, threshold: float=0.5):
        self.predict_probs = self.__get_predict(input_text)
        preds = list(map(lambda p: 0 if p < threshold else 1, self.predict_probs))

        return preds
    
    def predict_proba_profanity(self, input_text: str):
        self.predict_probs = self.__get_predict(input_text)

        return self.predict_probs

    def analyze_toxicity(self,text: str, aggregate: bool =True, return_proba: bool=True, threshold: float = 0.5):
        """
        Calculate toxicity of a text (if aggregate=True) or a vector of toxicity aspects (if aggregate=False). If return_proba = True, return real number - probability of text ot be toxic.
        If return_proba = False, returns class of a text(1 of toxic, 0 of non-toxic). Threshould should be from 0 to 1, this param controls border to mark text as toxic(if probability > threshould text is toxic).

        ATTENTION
            The model for predictions got from here https://huggingface.co/cointegrated/rubert-tiny-toxicity
        """
        print(text)
        text = ' '.join(self.__text_preparator.prepare_text(text, basing=False))
        print(text)
        with torch.no_grad():
            inputs = self.__tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(self.__model_toxicity.device)
            proba = torch.sigmoid(self.__model_toxicity(**inputs).logits).cpu().numpy()
            
        if isinstance(text, str):
            proba = proba[0]

        if return_proba:
            if aggregate:
                return 1 - proba.T[0] * (1 - proba.T[-1])
            
            return proba
        else:
            print((1 - proba.T[0] * (1 - proba.T[-1])))
            return int((1 - proba.T[0] * (1 - proba.T[-1])) > threshold)

    def __get_predict(self, input_text):
        input_text = self.__text_preparator.prepare_text(input_text, word_basing_method='stemming')
        vec_text = self.__vectorizer.transform(input_text)
        probabilities = self.__model_profanity.predict_proba(vec_text)[:, 1]

        return probabilities

analyzer = TextAnalyzer()

app = FastAPI()

# Костыль, не знаю как сделать по-другому
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)


@app.get("/analyze")
def analyze_text_get(text: str = Query(..., description="Text to analyze"), threshold: float = 0.12):
    """
    Analyze text toxicity via GET request.
    Parameters:
        text: str - text, that should be analyzed.
        threshold: float - threshold for mark text as a toxic.
    Returns:
        text: text which was analyzed, label: toxicity label, profanity: do text has or not profanity words. 
    """
    class_ = analyzer.analyze_toxicity(text, return_proba=False, threshold=threshold)
    has_profanity = any(i for i in analyzer.predict_profanity(text))
    return {"text": text, "label": 'Текст не ок' if class_ == 1 else 'Текст ок', "profanity" : "Есть маты" if has_profanity else "Матов нет"}

