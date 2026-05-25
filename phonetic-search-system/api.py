from fastapi import FastAPI
from pydantic import BaseModel
from src.search_engine import PhoneticSearchEngine

# Initialize API
app = FastAPI(title="Phonetic Search API")

# Load saved model (only once on startup)
engine = PhoneticSearchEngine()
print("Loading saved models...")
engine.load_all("models/phonetic")
print("Models loaded! API ready.")

class SearchRequest(BaseModel):
    query: str
    top_k: int = 10

@app.get("/")
def home():
    return {"message": "Phonetic Search API", "status": "ready"}

@app.post("/search")
def search(request: SearchRequest):
    results = engine.search(request.query, top_k=request.top_k)
    return {"query": request.query, "results": results}

@app.get("/stats")
def stats():
    return {
        "total_words": len(engine.words),
        "model": "TF-IDF + FAISS",
        "accuracy": "85.25% @Top-1"
    }