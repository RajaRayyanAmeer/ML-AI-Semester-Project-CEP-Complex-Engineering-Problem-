"""Phonetic encoding strategies (Single Responsibility)"""
from abc import ABC, abstractmethod
import fuzzy

class PhoneticEncoder(ABC):
    """Abstract base for encoders (Interface Segregation)"""
    @abstractmethod
    def encode(self, text: str) -> str:
        pass

class SoundexEncoder(PhoneticEncoder):
    def __init__(self):
        self.soundex = fuzzy.Soundex(4)
    
    def encode(self, text: str) -> str:
        return self.soundex(text)

class MetaphoneEncoder(PhoneticEncoder):
    def __init__(self):
        self.dmeta = fuzzy.DMetaphone()
    
    def encode(self, text: str) -> str:
        result = self.dmeta(text)
        return result[0] if result else ""

class PhonemeEncoder(PhoneticEncoder):
    """Direct phoneme-based encoding"""
    def __init__(self, cmu_dict: dict):
        self.cmu_dict = cmu_dict
    
    def encode(self, text: str) -> str:
        return self.cmu_dict.get(text.lower(), "")