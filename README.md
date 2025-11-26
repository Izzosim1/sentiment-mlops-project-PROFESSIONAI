# MachineInnovators Inc. - Monitoraggio Reputazione Online

Sistema MLOps per l'analisi automatica del sentiment sui social media, sviluppato per monitorare e migliorare la reputazione aziendale.

## Made by Simone Izzo per progetto Modulo Mlops ProfessionAI

## Indice

- [Obiettivo del Progetto](#obiettivo-del-progetto)
- [Tecnologie Utilizzate](#tecnologie-utilizzate)
- [Struttura del Repository](#struttura-del-repository)
- [Installazione](#installazione)
- [Utilizzo](#utilizzo)
- [Pipeline CI/CD](#pipeline-cicd)
- [Risultati](#risultati)
- [Demo](#demo)
- [Scelte Progettuali](#scelte-progettuali)

## Obiettivo del Progetto

Il progetto implementa un sistema MLOps completo per:

1. Analisi automatica del sentiment: classificazione di testi dai social media in positivo, neutro o negativo
2. Monitoraggio continuo: tracciamento delle predizioni e analisi della distribuzione del sentiment nel tempo
3. Pipeline CI/CD*: automazione di test, valutazione e deploy del modello

## Tecnologie Utilizzate

| Componente | Tecnologia |
|------------|------------|
| Modello NLP | RoBERTa (twitter-roberta-base-sentiment-latest) |
| Framework ML | HuggingFace Transformers, PyTorch |
| Web App | Gradio |
| CI/CD | GitHub Actions |
| Testing | Pytest |
| Dataset | TweetEval (HuggingFace Datasets) |

## Struttura del Repository
```
sentiment-mlops-project/
├── .github/
│   └── workflows/
│       └── ci-cd.yml          # Pipeline CI/CD
├── src/
│   ├── __init__.py
│   ├── model.py               # Classe SentimentAnalyzer
│   ├── evaluate.py            # Script di valutazione
│   └── monitoring.py          # Sistema di monitoraggio
├── tests/
│   ├── __init__.py
│   └── test_model.py          # Test unitari (9 test)
├── logs/
│   └── predictions.jsonl      # Log delle predizioni
├── metrics/
│   └── evaluation_results.json # Risultati valutazione
├── app.py                     # Applicazione Gradio
├── requirements.txt           # Dipendenze
└── README.md                  # Documentazione
```

## Installazione
```bash
# Clona il repository
git clone https://github.com/Izzosim1/sentiment-mlops-project-PROFESSIONAI.git
cd sentiment-mlops-project-PROFESSIONAI

# Installa le dipendenze
pip install -r requirements.txt
```

## Utilizzo

### Avviare l'applicazione web
```bash
python app.py
```

L'app sarà disponibile sul link che apparirà nel terminale

### Eseguire i test
```bash
pytest tests/ -v
```

### Valutare il modello
```bash
python -m src.evaluate
```

### Generare report di monitoraggio
```bash
python -m src.monitoring
```

## Pipeline CI/CD

La pipeline GitHub Actions esegue automaticamente a ogni push:

1. Test: esecuzione di 9 test unitari con pytest
2. Evaluate: valutazione del modello sul dataset TweetEval
3. Build: preparazione per il deploy

La pipeline è definita in `.github/workflows/ci-cd.yml`.

## Risultati

### Performance del Modello

Valutazione su 500 campioni del dataset TweetEval:

| Metrica | Valore |
|---------|--------|
| Accuracy | 69.8% |
| F1-Score (macro) | 70.5% |

### Classification Report

| Classe | Precision | Recall | F1-Score |
|--------|-----------|--------|----------|
| Negativo | 0.643 | 0.793 | 0.710 |
| Neutro | 0.733 | 0.617 | 0.670 |
| Positivo | 0.726 | 0.745 | 0.735 |

### Confusion Matrix

|  | Pred. Neg | Pred. Neu | Pred. Pos |
|--|-----------|-----------|-----------|
| **Actual Neg** | 119 | 29 | 2 |
| **Actual Neu** | 63 | 148 | 29 |
| **Actual Pos** | 3 | 25 | 82 |

## Demo

L'applicazione è deployata su HuggingFace Spaces:

[https://huggingface.co/spaces/Izzosim1/sentiment-analysis-mlops-project-ProfessioAI]

## Scelte Progettuali

### Modello: RoBERTa 

Il progetto originale suggeriva FastText, ma ho optato per RoBERTa(twitter-roberta-base-sentiment-latest) per i seguenti motivi:

1. Accuratezza superiore: i modelli transformer superano FastText nella comprensione del contesto
2. Pre-addestramento specifico: il modello è addestrato su 124M tweet, ideale per il dominio social media
3. Disponibilità su HuggingFace: integrazione nativa con l'ecosistema Transformers

### Dataset: TweetEval

Abbiamo scelto TweetEval perché:

1. Benchmark standard: utilizzato per valutare modelli NLP su Twitter
2. Qualità delle etichette: annotazioni verificate manualmente
3. Dimensione adeguata: oltre 45.000 tweet per training, 12.000 per test

### Preprocessing

Il preprocessing normalizza:
- Menzioni (@username → @user): protegge la privacy e standardizza l'input
- URL (http://... → http): rimuove variabilità non informativa

### Sistema di Monitoraggio

Il monitoraggio traccia:
- Timestamp: quando è avvenuta la predizione
- Testo: primi 100 caratteri per privacy
- Probabilità: confidence per ogni classe
- Sentiment dominante: classificazione finale

Questo permette di:
- Analizzare trend temporali
- Rilevare drift nella distribuzione del sentiment
- Valutare la confidenza media del modello

### Pipeline CI/CD

La pipeline è strutturata in tre job sequenziali:
1. Test first: i test devono passare prima di procedere
2. Evaluate: valutazione automatica delle performance
3. Build: solo su push a main, non su pull request

Questo garantisce che solo codice testato e valutato raggiunga la produzione.

