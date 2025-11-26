import pytest
from src.model import SentimentAnalyzer


@pytest.fixture(scope="module")
def analyzer():
    """Carica il modello una sola volta per tutti i test."""
    return SentimentAnalyzer()


def test_model_loads(analyzer):
    """Verifica che il modello si carichi correttamente."""
    assert analyzer.model is not None
    assert analyzer.tokenizer is not None


def test_prediction_returns_dict(analyzer):
    """Verifica che predict restituisca un dizionario."""
    result = analyzer.predict("test text")
    assert isinstance(result, dict)


def test_prediction_has_all_labels(analyzer):
    """Verifica che il risultato contenga tutte le etichette."""
    result = analyzer.predict("test text")
    assert 'positivo' in result
    assert 'negativo' in result
    assert 'neutro' in result


def test_probabilities_sum_to_one(analyzer):
    """Verifica che le probabilità sommino a 1."""
    result = analyzer.predict("test text")
    total = sum(result.values())
    assert abs(total - 1.0) < 0.01


def test_positive_sentiment(analyzer):
    """Verifica che frasi positive siano classificate correttamente."""
    result = analyzer.predict("I love this, it's absolutely wonderful!")
    assert result['positivo'] > result['negativo']


def test_negative_sentiment(analyzer):
    """Verifica che frasi negative siano classificate correttamente."""
    result = analyzer.predict("I hate this, it's terrible and awful")
    assert result['negativo'] > result['positivo']


def test_preprocessing_mentions(analyzer):
    """Verifica che le menzioni vengano sostituite."""
    processed = analyzer.preprocess("Hello @john_doe how are you")
    assert '@user' in processed


def test_preprocessing_urls(analyzer):
    """Verifica che gli URL vengano sostituiti."""
    processed = analyzer.preprocess("Check this https://example.com/page")
    assert 'http' in processed


def test_get_sentiment_label(analyzer):
    """Verifica che get_sentiment_label restituisca una stringa valida."""
    label = analyzer.get_sentiment_label("I love this!")
    assert label in ['positivo', 'negativo', 'neutro']