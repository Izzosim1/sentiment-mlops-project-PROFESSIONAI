from datasets import load_dataset
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import json
import os
from src.model import SentimentAnalyzer


def load_tweeteval_sentiment():
    """Carica il dataset TweetEval per sentiment analysis."""
    dataset = load_dataset("tweet_eval", "sentiment")
    return dataset


def map_prediction_to_label(prediction):
    """Converte la predizione del modello in etichetta numerica."""
    label_map = {'negativo': 0, 'neutro': 1, 'positivo': 2}
    dominant = max(prediction, key=prediction.get)
    return label_map[dominant]


def evaluate_model(analyzer, dataset_split, max_samples=None):
    """Valuta il modello su un split del dataset."""
    texts = dataset_split['text']
    true_labels = dataset_split['label']
    
    if max_samples:
        texts = texts[:max_samples]
        true_labels = true_labels[:max_samples]
    
    predicted_labels = []
    
    print(f"Valutazione su {len(texts)} campioni...")
    
    for i, text in enumerate(texts):
        if (i + 1) % 100 == 0:
            print(f"  Processato {i + 1}/{len(texts)}")
        
        prediction = analyzer.predict(text)
        pred_label = map_prediction_to_label(prediction)
        predicted_labels.append(pred_label)
    
    accuracy = accuracy_score(true_labels, predicted_labels)
    report = classification_report(
        true_labels, 
        predicted_labels,
        target_names=['negativo', 'neutro', 'positivo'],
        output_dict=True
    )
    cm = confusion_matrix(true_labels, predicted_labels)
    
    results = {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'num_samples': len(texts)
    }
    
    return results


def print_results(results):
    """Stampa i risultati in formato leggibile."""
    print("\n" + "=" * 50)
    print("RISULTATI VALUTAZIONE")
    print("=" * 50)
    print(f"\nAccuracy: {results['accuracy']:.4f}")
    print(f"Campioni valutati: {results['num_samples']}")
    
    print("\nClassification Report:")
    print("-" * 50)
    
    report = results['classification_report']
    print(f"{'Classe':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 50)
    
    for label in ['negativo', 'neutro', 'positivo']:
        metrics = report[label]
        print(f"{label:<12} {metrics['precision']:<12.3f} {metrics['recall']:<12.3f} {metrics['f1-score']:<12.3f}")
    
    print("-" * 50)
    print(f"{'macro avg':<12} {report['macro avg']['precision']:<12.3f} {report['macro avg']['recall']:<12.3f} {report['macro avg']['f1-score']:<12.3f}")
    
    print("\nConfusion Matrix:")
    print("              Predicted")
    print("              neg    neu    pos")
    cm = results['confusion_matrix']
    labels = ['neg', 'neu', 'pos']
    for i, row in enumerate(cm):
        print(f"Actual {labels[i]:>3}   {row[0]:<6} {row[1]:<6} {row[2]:<6}")


def save_results(results, output_dir="metrics"):
    """Salva i risultati in formato JSON."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "evaluation_results.json")
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nRisultati salvati in: {output_path}")


if __name__ == "__main__":
    print("Caricamento modello...")
    analyzer = SentimentAnalyzer()
    
    print("Caricamento dataset TweetEval...")
    dataset = load_tweeteval_sentiment()
    
    results = evaluate_model(
        analyzer, 
        dataset['test'], 
        max_samples=500
    )
    
    print_results(results)
    save_results(results)