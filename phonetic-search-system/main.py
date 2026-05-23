"""Main entry point"""
from src.search_engine import PhoneticSearchEngine
from src.phonetic_encoder import SoundexEncoder, MetaphoneEncoder, PhonemeEncoder
from src.utils import download_cmudict

def main():
    # Download data
    print("Downloading CMU Dictionary...")
    data_path = download_cmudict()
    
    # Initialize engine
    print("\nInitializing search engine...")
    engine = PhoneticSearchEngine()
    
    # Load and build
    engine.load_data(data_path)
    engine.build_index()
    
    # Interactive search
    print("Phonetic Search System Ready!")
    
    while True:
        query = input("\nEnter word (or 'quit'): ").strip()
        if query.lower() == 'quit':
            break
    
        # Check if word exists
        if query.lower() not in engine.cmu_dict:
            print(f" '{query}' not found in dictionary. Showing approximate matches:")
    
        results = engine.search(query, top_k=10)
    
        # Check if results are valid
        if results and results[0]['distance'] >= 0.9:
            print(f" No phonetic matches found for '{query}'")
            continue
    
    print(f"\nTop matches for '{query}':")
    for i, res in enumerate(results, 1):
        print(f"{i}. {res['word']} (distance: {res['distance']:.4f})")

if __name__ == "__main__":
    main()