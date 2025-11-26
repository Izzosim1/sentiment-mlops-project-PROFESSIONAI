import json
import os
from collections import Counter
from datetime import datetime, timedelta


class SentimentMonitor:
    """Sistema di monitoraggio per le predizioni del sentiment."""
    
    def __init__(self, log_file="logs/predictions.jsonl"):
        self.log_file = log_file
    
    def load_logs(self, hours=24):
        """Carica i log delle ultime N ore."""
        if not os.path.exists(self.log_file):
            return []
        
        cutoff = datetime.now() - timedelta(hours=hours)
        logs = []
        
        with open(self.log_file, 'r') as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    entry_time = datetime.fromisoformat(entry['timestamp'])
                    if entry_time > cutoff:
                        logs.append(entry)
        return logs
    
    def get_sentiment_distribution(self, hours=24):
        """Calcola la distribuzione del sentiment."""
        logs = self.load_logs(hours)
        
        if not logs:
            return {"positivo": 0, "neutro": 0, "negativo": 0}
        
        sentiments = [log['dominant_sentiment'] for log in logs]
        counts = Counter(sentiments)
        
        return dict(counts)
    
    def get_statistics(self, hours=24):
        """Calcola statistiche generali."""
        logs = self.load_logs(hours)
        
        if not logs:
            return {
                "total_predictions": 0,
                "distribution": {},
                "avg_confidence": 0
            }
        
        distribution = self.get_sentiment_distribution(hours)
        
        # Calcola confidenza media
        confidences = []
        for log in logs:
            pred = log['prediction']
            dominant = log['dominant_sentiment']
            confidences.append(pred[dominant])
        
        avg_confidence = sum(confidences) / len(confidences)
        
        return {
            "total_predictions": len(logs),
            "distribution": distribution,
            "avg_confidence": avg_confidence,
            "period_hours": hours
        }
    
    def detect_sentiment_shift(self, baseline_hours=168, recent_hours=24, threshold=0.15):
        """
        Rileva cambiamenti significativi nel sentiment.
        Confronta le ultime recent_hours con il baseline.
        """
        baseline = self.get_sentiment_distribution(baseline_hours)
        recent = self.get_sentiment_distribution(recent_hours)
        
        baseline_total = sum(baseline.values()) or 1
        recent_total = sum(recent.values()) or 1
        
        shifts = {}
        alert = False
        
        for sentiment in ['positivo', 'neutro', 'negativo']:
            baseline_pct = baseline.get(sentiment, 0) / baseline_total
            recent_pct = recent.get(sentiment, 0) / recent_total
            shift = recent_pct - baseline_pct
            shifts[sentiment] = shift
            
            if abs(shift) > threshold:
                alert = True
        
        return {
            "alert": alert,
            "shifts": shifts,
            "baseline_period": baseline_hours,
            "recent_period": recent_hours
        }
    
    def generate_report(self):
        """Genera un report completo."""
        stats_24h = self.get_statistics(24)
        stats_7d = self.get_statistics(168)
        shift = self.detect_sentiment_shift()
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "last_24_hours": stats_24h,
            "last_7_days": stats_7d,
            "sentiment_shift": shift
        }
        
        return report


def print_report(report):
    """Stampa il report in formato leggibile."""
    print("\n" + "=" * 50)
    print("REPORT MONITORAGGIO SENTIMENT")
    print("=" * 50)
    print(f"Generato: {report['generated_at']}")
    
    print("\n--- Ultime 24 ore ---")
    stats = report['last_24_hours']
    print(f"Predizioni totali: {stats['total_predictions']}")
    if stats['total_predictions'] > 0:
        print(f"Confidenza media: {stats['avg_confidence']:.2%}")
        print("Distribuzione:")
        for sent, count in stats['distribution'].items():
            print(f"  {sent}: {count}")
    
    print("\n--- Ultimi 7 giorni ---")
    stats = report['last_7_days']
    print(f"Predizioni totali: {stats['total_predictions']}")
    
    print("\n--- Rilevamento Shift ---")
    shift = report['sentiment_shift']
    if shift['alert']:
        print("ALERT: Rilevato cambiamento significativo nel sentiment!")
    else:
        print("Nessun cambiamento significativo rilevato.")


if __name__ == "__main__":
    monitor = SentimentMonitor()
    report = monitor.generate_report()
    print_report(report)