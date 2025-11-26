from src.model import SentimentAnalyzer

print("Caricamento modello...")
analyzer = SentimentAnalyzer()
print("Modello caricato.\n")

test_phrases = [
    "I love this product, it's amazing!",
    "This is the worst experience ever",
    "The weather is nice today",
    "Il servizio clienti è stato eccellente",
    "Non comprerò mai più da questo negozio"
]

for phrase in test_phrases:
    result = analyzer.predict(phrase)
    dominant = analyzer.get_sentiment_label(phrase)
    
    print(f"Testo: {phrase}")
    print(f"Sentiment: {dominant}")
    print(f"Prob: pos={result['positivo']:.3f}, neu={result['neutro']:.3f}, neg={result['negativo']:.3f}")
    print("-" * 50)