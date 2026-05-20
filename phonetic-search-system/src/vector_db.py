"""Vector database operations (Single Responsibility)"""
import faiss
import numpy as np

class VectorDatabase:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.word_map = []
    
    def add_vectors(self, vectors: np.ndarray, words: list):
        """Add vectors to index"""
        self.index.add(vectors)
        self.word_map.extend(words)
    
    def search(self, query_vector: np.ndarray, top_k: int = 10):
        """Search similar vectors"""
        distances, indices = self.index.search(query_vector, top_k)
        results = [
            {'word': self.word_map[idx], 'distance': float(dist)}
            for dist, idx in zip(distances[0], indices[0])
            if idx < len(self.word_map)
        ]
        return results
    
    def save(self, path: str):
        """Save index"""
        faiss.write_index(self.index, path)
    
    def load(self, path: str):
        """Load index"""
        self.index = faiss.read_index(path)