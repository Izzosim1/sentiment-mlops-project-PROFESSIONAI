import gradio as gr
import json
import os
from datetime import datetime
from src.model import SentimentAnalyzer

# Inizializza il modello
print("Caricamento modello...")
analyzer = SentimentAnalyzer()
print("Modello caricato.")

# Crea cartella logs se non esiste
os.makedirs("logs", exist_ok=True)


def log_prediction(text, result):
    """Salva la predizione nel file di log per monitoraggio."""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "text": text[:100],  # Troncato per privacy
        "prediction": result,
        "dominant_sentiment": max(result, key=result.get)
    }
    
    log_path = "logs/predictions.jsonl"
    with open(log_path, "a") as f:
        f.write(json.dumps(log_entry) + "\n")


def analyze_sentiment(text):
    """Funzione principale per l'analisi del sentiment."""
    if not text or text.strip() == "":
        return {
            "Positivo": 0.0,
            "Neutro": 0.0,
            "Negativo": 0.0
        }
    
    # Ottieni predizione
    result = analyzer.predict(text)
    
    # Log per monitoraggio
    log_prediction(text, result)
    
    # Formatta output per Gradio
    return {
        "Positivo": result['positivo'],
        "Neutro": result['neutro'],
        "Negativo": result['negativo']
    }


def analyze_batch(texts):
    """Analizza più testi separati da newline."""
    if not texts or texts.strip() == "":
        return "Inserisci almeno un testo."
    
    lines = [line.strip() for line in texts.split("\n") if line.strip()]
    results = []
    
    for line in lines:
        pred = analyzer.predict(line)
        dominant = max(pred, key=pred.get)
        confidence = pred[dominant]
        results.append(f"- \"{line[:50]}...\" → {dominant.upper()} ({confidence:.1%})")
    
    return "\n".join(results)


# Interfaccia Gradio
with gr.Blocks(title="Sentiment Analysis - MLOps Demo") as demo:
    gr.Markdown("""
    # Analisi del Sentiment per Social Media
    
    Questo strumento analizza il sentiment di testi dai social media, 
    classificandoli come **positivo**, **neutro** o **negativo**.
    
    Modello: [twitter-roberta-base-sentiment](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)
    """)
    
    with gr.Tab("Analisi Singola"):
        with gr.Row():
            with gr.Column():
                input_text = gr.Textbox(
                    label="Inserisci il testo",
                    placeholder="Scrivi qui il testo da analizzare...",
                    lines=3
                )
                analyze_btn = gr.Button("Analizza", variant="primary")
            
            with gr.Column():
                output_label = gr.Label(label="Risultato Sentiment")
        
        analyze_btn.click(
            fn=analyze_sentiment,
            inputs=input_text,
            outputs=output_label
        )
        
        gr.Examples(
            examples=[
                ["I love this product, it's amazing!"],
                ["This is the worst experience ever, totally disappointed"],
                ["Just bought a new phone, it's okay I guess"],
                ["The customer service was incredibly helpful and friendly"],
                ["Waiting in line for 2 hours, this is ridiculous"]
            ],
            inputs=input_text
        )
    
    with gr.Tab("Analisi Batch"):
        gr.Markdown("Inserisci più testi, uno per riga.")
        batch_input = gr.Textbox(
            label="Testi (uno per riga)",
            placeholder="Primo testo...\nSecondo testo...\nTerzo testo...",
            lines=6
        )
        batch_btn = gr.Button("Analizza Tutti", variant="primary")
        batch_output = gr.Textbox(label="Risultati", lines=8)
        
        batch_btn.click(
            fn=analyze_batch,
            inputs=batch_input,
            outputs=batch_output
        )

# Avvia l'app
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)