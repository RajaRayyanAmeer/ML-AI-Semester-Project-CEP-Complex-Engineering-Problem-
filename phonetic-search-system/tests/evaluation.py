"""Full dataset evaluation"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.search_engine import PhoneticSearchEngine
import pandas as pd
import numpy as np
import random
import time

def full_dataset_evaluation(engine, sample_size=10000):
    """Evaluate on full dataset"""
    print(f"\nEvaluating on {sample_size} random samples...")
    
    random_words = random.sample(engine.words, sample_size)
    
    metrics = {
        'retrieval_top1': 0,
        'retrieval_top5': 0,
        'retrieval_top10': 0,
        'avg_distance_top1': [],
        'avg_time_ms': []
    }
    
    for word in random_words:
        start = time.time()
        results = engine.search(word, top_k=10)
        elapsed = (time.time() - start) * 1000
        
        top_words = [r['word'] for r in results]
        
        # Check if exact match in top-k
        if word in top_words[:1]:
            metrics['retrieval_top1'] += 1
        if word in top_words[:5]:
            metrics['retrieval_top5'] += 1
        if word in top_words[:10]:
            metrics['retrieval_top10'] += 1
        
        metrics['avg_distance_top1'].append(results[0]['distance'])
        metrics['avg_time_ms'].append(elapsed)
    
    # Calculate final metrics
    results_df = pd.DataFrame({
        'Metric': [
            'Retrieval Accuracy @1',
            'Retrieval Accuracy @5',
            'Retrieval Accuracy @10',
            'Avg Distance (Top-1)',
            'Avg Search Time (ms)',
            'Total Words Tested',
            'Database Size'
        ],
        'Value': [
            f"{metrics['retrieval_top1']/sample_size*100:.2f}%",
            f"{metrics['retrieval_top5']/sample_size*100:.2f}%",
            f"{metrics['retrieval_top10']/sample_size*100:.2f}%",
            f"{np.mean(metrics['avg_distance_top1']):.4f}",
            f"{np.mean(metrics['avg_time_ms']):.2f}",
            sample_size,
            len(engine.words)
        ]
    })
    
    print("\n" + "="*50)
    print("FULL DATASET EVALUATION RESULTS")
    print("="*50)
    print(results_df.to_string(index=False))
    
    # Save
    results_df.to_csv('results/full_evaluation.csv', index=False)
    print("\nSaved to: results/full_evaluation.csv")

if __name__ == "__main__":
    engine = PhoneticSearchEngine()
    engine.load_data("data/cmudict-0.7b.txt")
    engine.build_index()
    full_dataset_evaluation(engine, sample_size=10000)