"""Utility functions"""
import requests
import pickle
from pathlib import Path

def download_cmudict(output_path: str = "data/cmudict-0.7b.txt"):
    """Download CMU dictionary"""
    url = "https://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict-0.7b"
    Path(output_path).parent.mkdir(exist_ok=True)
    response = requests.get(url)
    Path(output_path).write_bytes(response.content)
    return output_path

def save_model(obj, path: str):
    """Save model to disk"""
    Path(path).parent.mkdir(exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_model(path: str):
    """Load model from disk"""
    with open(path, 'rb') as f:
        return pickle.load(f)

def parse_cmudict(file_path: str) -> dict:
    """Parse CMU dictionary into word->phonemes mapping"""
    word_phonemes = {}
    with open(file_path, 'r', encoding='latin-1') as f:
        for line in f:
            if line.startswith(';;;') or not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            word = parts[0].split('(')[0].lower()
            phonemes = ' '.join(parts[1:])
            word_phonemes[word] = phonemes
    return word_phonemes