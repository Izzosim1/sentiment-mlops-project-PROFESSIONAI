from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
from scipy.special import softmax


class SentimentAnalyzer:
    """
    Classe per l'analisi del sentiment usando il modello RoBERTa
    pre-addestrato su Twitter.
    """
    
    def __init__(self, model_name="cardiffnlp/twitter-roberta-base-sentiment-latest"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.labels = ['negativo', 'neutro', 'positivo']
    
    def preprocess(self, text):
        """
        Sostituisce menzioni e URL con token standard.
        """
        new_text = []
        for token in text.split(" "):
            if token.startswith('@') and len(token) > 1:
                token = '@user'
            elif token.startswith('http'):
                token = 'http'
            new_text.append(token)
        return " ".join(new_text)
    
    def predict(self, text):
        """
        Restituisce le probabilità per ogni classe di sentiment.
        """
        processed_text = self.preprocess(text)
        
        encoded_input = self.tokenizer(
            processed_text,
            return_tensors='pt',
            truncation=True,
            max_length=512
        )
        
        output = self.model(**encoded_input)
        scores = output.logits[0].detach().numpy()
        probabilities = softmax(scores)
        
        result = {}
        for i, label in enumerate(self.labels):
            result[label] = float(probabilities[i])
        
        return result
    
    def get_sentiment_label(self, text):
        """
        Restituisce solo l'etichetta del sentiment dominante.
        """
        predictions = self.predict(text)
        return max(predictions, key=predictions.get)