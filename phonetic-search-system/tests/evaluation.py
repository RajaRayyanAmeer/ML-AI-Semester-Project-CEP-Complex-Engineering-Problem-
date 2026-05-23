"""Evaluation script for phonetic search"""
from src.search_engine import PhoneticSearchEngine
from src.utils import download_cmudict
import pandas as pd
import time

# Test cases: (query, expected_matches)
TEST_CASES = [
    ("there", ["their", "they're", "there"]),
    ("night", ["knight", "nite"]),
    ("write", ["right", "rite", "wright"]),
    ("sea", ["see", "c"]),
    ("two", ["too", "to"]),
    ("flower", ["flour"]),
    ("bear", ["bare", "beer"]),
    ("peace", ["piece"]),
    ("blue", ["blew"]),
    ("brake", ["break"]),
]

def evaluate_hybrid(engine):
    """1. Phonetic pairs (quality) + 2. Random sampling (coverage)"""
    
    # Part 1: Phonetic quality (homophones)
    phonetic_pairs = [("there", ["their"]), ("night", ["knight"]), ...]
    
    # Part 2: Random retrieval test
    import random
    random_words = random.sample(engine.words, 100)
    
    # Test if system retrieves exact match in top-10
    retrieval_success = sum(
        1 for word in random_words 
        if word in [r['word'] for r in engine.search(word, 10)]
    ) / 100
    
    print(f"Retrieval Rate: {retrieval_success:.2%}")

def precision_at_k(results, expected, k=10):
    """Calculate precision@k"""
    top_k = [r['word'] for r in results[:k]]
    matches = sum(1 for word in expected if word in top_k)
    return matches / len(expected) if expected else 0

def evaluate_engine(engine):
    """Run evaluation"""
    results_data = []
    
    for query, expected in TEST_CASES:
        start = time.time()
        results = engine.search(query, top_k=10)
        elapsed = time.time() - start
        
        p5 = precision_at_k(results, expected, k=5)
        p10 = precision_at_k(results, expected, k=10)
        
        results_data.append({
            'query': query,
            'expected': ', '.join(expected),
            'precision@5': p5,
            'precision@10': p10,
            'time_ms': elapsed * 1000
        })
        
        print(f"{query}: P@5={p5:.2f}, P@10={p10:.2f}")
    
    df = pd.DataFrame(results_data)
    df.to_csv('results/evaluation.csv', index=False)
    
    print(f"\nAverage Precision@5: {df['precision@5'].mean():.2f}")
    print(f"Average Precision@10: {df['precision@10'].mean():.2f}")
    print(f"Average Time: {df['time_ms'].mean():.2f}ms")

if __name__ == "__main__":
    engine = PhoneticSearchEngine()
    engine.load_data("data/cmudict-0.7b.txt")
    engine.build_index()
    evaluate_engine(engine)
    evaluate_hybrid(engine)