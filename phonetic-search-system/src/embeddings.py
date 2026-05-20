"""Embedding generation (Single Responsibility)"""
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class EmbeddingGenerator:
    def __init__(self, method='tfidf'):
        self.method = method
        self.vectorizer = TfidfVectorizer(
            analyzer='char',
            ngram_range=(2, 4),
            max_features=500
        )
        self.is_fitted = False
    
    def fit(self, texts: list):
        """Fit on corpus"""
        self.vectorizer.fit(texts)
        self.is_fitted = True
    
    def transform(self, texts: list) -> np.ndarray:
        """Generate embeddings"""
        if not self.is_fitted:
            raise ValueError("Must fit before transform")
        return self.vectorizer.transform(texts).toarray().astype('float32')
    
    def fit_transform(self, texts: list) -> np.ndarray:
        """Fit and transform"""
        self.fit(texts)
        return self.transform(texts)