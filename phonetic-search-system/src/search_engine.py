"""Main search engine (Dependency Inversion)"""
from sys import prefix

from .phonetic_encoder import PhoneticEncoder, SoundexEncoder, MetaphoneEncoder, PhonemeEncoder
from .embeddings import EmbeddingGenerator
from .vector_db import VectorDatabase
from .utils import parse_cmudict
from .utils import save_model, load_model
import numpy as np

class PhoneticSearchEngine:
    """Main orchestrator following Open/Closed principle"""
    
    def __init__(self, encoder: PhoneticEncoder = None):
        self.encoder = encoder or PhonemeEncoder({})
        self.embedding_gen = EmbeddingGenerator()
        self.vector_db = None
        self.cmu_dict = {}
        self.words = []
    
    def load_data(self, dict_path: str):
        """Load and parse dictionary"""
        self.cmu_dict = parse_cmudict(dict_path)
        self.words = list(self.cmu_dict.keys())
        self.encoder = PhonemeEncoder(self.cmu_dict)
        print(f"Loaded {len(self.words)} words")
    
    def build_index(self):
        """Build vector index"""
        print("Encoding words...")
        encoded = [self.encoder.encode(word) for word in self.words]
        
        print("Generating embeddings...")
        vectors = self.embedding_gen.fit_transform(encoded)
        
        print("Building FAISS index...")
        self.vector_db = VectorDatabase(dimension=vectors.shape[1])
        self.vector_db.add_vectors(vectors, self.words)
        print("Index built successfully!")
    
    def search(self, query: str, top_k: int = 10):
        """Search phonetically similar words"""
        if not self.vector_db:
            raise ValueError("Index not built. Call build_index() first")
        
        # Encode and embed query
        encoded_query = self.encoder.encode(query)
        query_vector = self.embedding_gen.transform([encoded_query])
        
        # Search
        results = self.vector_db.search(query_vector, top_k)
        return results
    
    def save_index(self, path: str = "models/faiss.index"):
        """Save trained index"""
        self.vector_db.save(path)
    
    def set_encoder(self, encoder: PhoneticEncoder):
        """Change encoding strategy (Strategy Pattern)"""
        self.encoder = encoder

    def save_all(self, prefix="models/phonetic"):
    self.vector_db.save(f"{prefix}_faiss.index")
    save_model(self.embedding_gen, f"{prefix}_embeddings.pkl")
    save_model(self.words, f"{prefix}_words.pkl")

def load_all(self, prefix="models/phonetic"):
    self.vector_db = VectorDatabase(500)
    self.vector_db.load(f"{prefix}_faiss.index")
    self.embedding_gen = load_model(f"{prefix}_embeddings.pkl")
    self.words = load_model(f"{prefix}_words.pkl")
    self.vector_db.word_map = self.words