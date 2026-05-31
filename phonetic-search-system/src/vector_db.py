"""Vector database operations (Single Responsibility)"""
from os import path

from importlib.resources import path

import faiss
import numpy as np
import os
import streamlit as st

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
    
    def load(self, path):
        """Load index with error handling for compatibility issues"""
        try:
            self.index = faiss.read_index(path)      # 1. Attempt to read the pre-built index file
    
        except Exception as e:
            st.warning(" Pre-built FAISS index is incompatible or corrupted. Rebuilding from raw data...")
        
            # 2. Find the raw features data to rebuild the index
            # We look for 'features.npy' inside the same models directory
            base_dir = os.path.dirname(path)
            features_path = os.path.join(base_dir, "features.npy")
        
            if os.path.exists(features_path):
                # Load your raw numpy embeddings
                vectors = np.load(features_path).astype('float32')
            
                # Use your vector_db's existing build method to initialize and add vectors
                self.build(vectors)
            
                # Save the newly built index on the Linux server so it's fast next time
                faiss.write_index(self.index, path)
                st.success(" FAISS Index rebuilt successfully from features.npy!")
        
            else:
                st.error(f" Critical Error: Could not find raw features data at {features_path} to rebuild index.")
                raise FileNotFoundError(f"Missing raw features file at {features_path}")