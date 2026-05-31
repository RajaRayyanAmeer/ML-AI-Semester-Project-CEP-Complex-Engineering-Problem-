# **ML-AI-Semester-Project-CEP-Complex-Engineering-Problem**
**About:** AI-powered phonetic similarity search system using vector embeddings and phonetic encoding algorithms to match words by pronunciation, not spelling.

**Phonetic Similarity Search System:** An intelligent search engine that finds words, names, and phrases based on how they sound rather than how they're spelled. Built using phonetic encoding (Soundex, Metaphone), ML embeddings, and FAISS vector database on the CMU Pronouncing Dictionary.

# ML&AI CEP: Phonetic Similarity Search System

**Course:** CP-310 Machine Learning & AI | **Semester:** 6 | **Entry:** 23-CP-25 | **Instructor:** Dr. Waqar Ahmad

A production-ready, AI-powered phonetic similarity search engine using CMU Pronouncing Dictionary, TF-IDF embeddings, and FAISS vector database. Built with SOLID principles and deployed on Streamlit Cloud.

## Key Metrics

| Metric | Value |
|--------|-------|
| **Dataset Size** | 125,067 words (CMU Dict 0.7b) |
| **Retrieval Accuracy @1** | **85.25%** |
| **Retrieval Accuracy @5** | **99.66%** |
| **Retrieval Accuracy @10** | **100.00%** |
| **Avg Search Time** | 44.86 ms |
| **FAISS Index Size** | 250 MB |
| **Model Dimensions** | 500 features (TF-IDF) |


## Live Deployment

**[Try Live Demo on Streamlit Cloud](https://phonetic-search-system-23-cp-25.streamlit.app/)**

## Core Components

| Component | Responsibility | Technology |
|-----------|-----------------|-------------|
| **PhoneticEncoder** | Text→Phoneme conversion | `fuzzy` lib (Soundex), Custom phoneme dict |
| **EmbeddingGenerator** | Text→Vector conversion | scikit-learn TF-IDF (char n-grams) |
| **VectorDatabase** | Vector storage & retrieval | FAISS IndexFlatL2 (L2 distance) |
| **SearchEngine** | Orchestration | Custom orchestrator with caching |


## Setup
- **Python:** 3.11+ (NOT 3.14 - see troubleshooting)
- **OS:** Windows, macOS, Linux
- **Storage:** ~500 MB (code + models + dependencies)
- **RAM:** 2 GB minimum (4 GB recommended for FAISS loading)

### Local Installation

#### 1. Clone Repository
```bash
git clone https://github.com/RajaRayyanAmeer/ML-AI-Semester-Project-CEP-Complex-Engineering-Problem-.git
cd ML-AI-Semester-Project-CEP-Complex-Engineering-Problem-
cd phonetic-search-system
```

#### 2. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

**What gets installed:**
- `streamlit` - Web UI framework
- `scikit-learn` - TF-IDF vectorization
- `faiss-cpu` - Vector similarity search
- `fuzzy` - Soundex/Metaphone encoding
- `pandas`, `numpy`, `matplotlib`, `seaborn` - Data processing & visualization

#### 4. Run Locally
```bash
streamlit run streamlit_app.py
```

Browser opens at `http://localhost:8501`

### REST AP
```bash
python api.py  # Runs FastAPI on http://localhost:8000
```

## Evaluation Results

### Accuracy Metrics
Evaluated on **10,000 random samples** from CMU Dictionary:

```
Retrieval Accuracy @1:   85.25%  (exact phonetic match found in top 1)
Retrieval Accuracy @5:   99.66%  (exact match found within top 5)
Retrieval Accuracy @10: 100.00%  (exact match always in top 10)
```

### Performance
```
Average Search Time:      44.86 ms
Min Search Time:          15.23 ms
Max Search Time:          187.45 ms
Query Throughput:         ~22 queries/second
```

### Phonetic Precision
```
Precision@10 (phonetically correct results): 90%
Mean Reciprocal Rank (MRR):                  0.87
```

See `results/evaluation.csv` for full breakdown.


## Troubleshooting & Deployment (A-to-Z)

### **ISSUE #1: Python 3.14 Incompatibility**

**Problem:**
Cloud platforms default to Python 3.14 (experimental). Critical packages (`faiss-cpu`, `gensim`, `numpy`) lack stable wheels for 3.14.

**Errors Encountered:**

faiss-cpu==1.7.4 has no wheels with a matching Python ABI tag
ModuleNotFoundError: No module named 'distutils'
gensim: Building wheel for gensim failed

**Root Cause:**
- Python 3.14 is alpha/experimental
- Binary wheels for numeric libraries only exist for LTS versions (3.11, 3.12)
- Attempting to compile from source hits deprecated C API symbols

**Solution Implemented:**
Create `.python-version` file in repo root:
```
3.11.0
```

**How It Works:**
- Streamlit Cloud reads `.python-version` and provisions Python 3.11 instead of 3.14
- Python 3.11 has pre-compiled binary wheels for all dependencies
- No source compilation needed
- LTS support ensures long-term stability

**Verification:**
```bash
python --version  # Should show 3.11.x
```

---

### **ISSUE #2: Unsupported Package Versions**

**Problem A: Gensim 4.4.0 Build Failure**

Error:
```
gensim/models/word2vec_inner.c(9364): error C2039: 'subarray': 
is not a member of '_PyArray_Descr'
```

**Why:** Gensim's Cython-compiled C extensions reference removed Python 3.14 C API symbols.

**Problem B: Faiss-cpu 1.7.4 No Wheels**

Error:
```
faiss-cpu==1.7.4 has no wheels with a matching Python ABI tag for Python 3.14
```

**Why:** Version 1.7.4 released before Python 3.14; newer versions have 3.14 support.

**Solution Implemented:**

Updated `requirements.txt` with compatible versions:

```txt
# BEFORE (BROKEN)
gensim>=4.3.0
faiss-cpu==1.7.4

# AFTER (FIXED)
faiss-cpu>=1.8.0        # Has Python 3.11+ wheels
# gensim REMOVED (see below)
```

**Full Fixed requirements.txt:**
```txt
streamlit>=1.58.0
scikit-learn>=1.4.0
pandas>=2.0.0
numpy>=1.26.0
faiss-cpu>=1.8.0
fuzzy>=1.2.0
matplotlib>=3.10.0
seaborn>=0.13.0
```

**Why Gensim Was Removed:**

Gensim is for:
- Word2Vec embeddings
- Doc2Vec document embeddings
- Topic modeling (LDA)
- Skip-gram, CBOW algorithms

**Our system uses:**
- ✅ `fuzzy` for Soundex/Metaphone phonetic encoding
- ✅ `scikit-learn TF-IDF` for character n-gram embeddings
- ✅ `FAISS` for vector similarity search
- ✅ Custom CMU dictionary phoneme encoder

**Finding:** Gensim never imported, never used, completely unnecessary.

**Result:** Removing it eliminated all build failures.

### **ISSUE #3: File Path Resolution Mismatch**

**Problem:**

Local code:
```python
engine.load_all("models/phonetic")
```

Local environment:
```
Working Dir: F:\...\phonetic-search-system\
Relative path: models/phonetic → F:\...\phonetic-search-system\models\phonetic
```

Cloud environment:
```
Working Dir: /mount/src/ml-ai-semester-project-cep-complex-engineering-problem-/
Repository structure: phonetic-search-system/
Path search: /mount/src/.../models/phonetic (NOT FOUND)
Actual location: /mount/src/.../phonetic-search-system/models/phonetic
```

**Error:**
```
FileNotFoundError: models/phonetic_faiss.index not found
RuntimeError: This app has encountered an error
```

**Root Cause:**
- Hardcoded relative paths assume execution from project root
- Cloud deployment changes working directory unpredictably
- Path resolution becomes ambiguous across Windows/Linux

**Solution Implemented:**

In `streamlit_app.py`:
```python
import os

@st.cache_resource
def load_engine():
    """Load engine with absolute path resolution"""
    
    # Get absolute path to THIS script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Build absolute path to models folder
    model_path = os.path.join(current_dir, "models", "phonetic")
    
    # Initialize and load engine
    engine = PhoneticSearchEngine()
    engine.load_all(model_path)
    
    return engine
```

**Why It Works:**

1. `os.path.abspath(__file__)` returns absolute path to `streamlit_app.py` regardless of where Python was invoked
2. `os.path.dirname()` extracts the containing directory
3. `os.path.join()` safely constructs paths cross-platform:
   - Windows: `C:\Users\...\models\phonetic`
   - Linux: `/mount/src/.../models/phonetic`
4. Result: Path is absolute, works everywhere

**Also Applied To:**
- `search_engine.py` `load_all()` method
- `vector_db.py` `load()` method

---

### **ISSUE #4: Large File Size Limitation (250 MB FAISS Index)**

**Problem:**

GitHub file size limit: **100 MB per file**
FAISS index size: **250 MB**

**Attempted Solution #1: Direct Push**
```bash
git add models/phonetic_faiss.index
git push origin main
```
**Failed:** `fatal: file too large (250 MB > 100 MB limit)`

**Attempted Solution #2: .gitignore Removal**
```bash
git add -f models/
git push origin main
```
**Failed:** Same 100 MB limit error

**Root Cause:**
GitHub enforces hard limit on individual file size. No workaround with standard Git.

**Solution Implemented: Git LFS (Large File Storage)**

#### Setup:
```bash
# 1. Install Git LFS client (if not already installed)
git lfs install

# 2. Track large files in your repo
git lfs track "models/phonetic_faiss.index"
git lfs track "models/*.pkl"

# 3. Commit .gitattributes (tells Git which files are LFS-tracked)
git add .gitattributes
git commit -m "Setup Git LFS for large model files"

# 4. Add and commit large files
git add models/
git commit -m "Add trained FAISS models via Git LFS"

# 5. Push (LFS automatically handles large files)
git push origin main
```

**How Git LFS Works:**

1. **Pointer File:** Git stores a tiny **50-byte pointer** in the repository:
   ```
   version https://git-lfs.github.com/spec/v1
   oid sha256:abc123...
   size 250000000
   ```

2. **Large File Storage:** Actual 250 MB file stored on LFS server (lfs.github.com)

3. **Transparent Download:** When cloning:
   ```bash
   git clone <repo>
   # Git automatically downloads pointer
   # LFS automatically downloads actual 250 MB file
   # User sees normal files
   ```

4. **Result:**
   - Repository size: ~5 MB (code + pointers)
   - LFS server: ~250 MB (large models)
   - Zero user friction

**Verification:**
```bash
git lfs ls-files

# Output:
# phonetic_faiss.index (250 MB)
# phonetic_embeddings.pkl (1.3 MB)
# phonetic_words.pkl (250 KB)
```

**Cloud Platform Support:**
-  Streamlit Cloud: Supports Git LFS (automatic download during build)
-  GitHub Actions: Supports Git LFS

### **ISSUE #5: Streamlit Caching & Performance**

**Problem:**

Streamlit reruns entire script on every user interaction:
- User types in search box → Full rerun
- User clicks button → Full rerun
- Loading 250 MB FAISS index on every rerun = **10+ seconds per interaction**

**Solution Implemented:**

Add Streamlit's caching decorator:
```python
@st.cache_resource
def load_engine():
    """Load once, reuse across all reruns"""
    print("Loading FAISS index (first time only)...")
    engine = PhoneticSearchEngine()
    engine.load_all(model_path)
    return engine

# First call: Loads index (10 seconds)
# Subsequent calls: Uses cache (instant)
engine = load_engine()
```

**How It Works:**

1. `@st.cache_resource` caches the loaded engine object
2. Decorator maintains cache across user interactions
3. One cache per user session
4. Cache invalidates only when:
   - Script file changes
   - Function code changes
   - Dependencies change

**Result:** Search now instant (<50ms)

---

### **ISSUE #6: Data Serialization Format (Windows vs Linux)**

**Problem:**

Models trained on **Windows**:
- Generated using Python 3.11 on Windows
- Serialized with pickle/FAISS binary format
- Potential endianness issues (Windows: little-endian)

Models deployed on **Linux** (Streamlit Cloud):
- Ubuntu Linux server
- Different OS-level memory layout
- Could cause deserialization failures

**Why It Could Fail:**
- Binary files contain platform-specific metadata
- Floating-point precision varies
- Memory alignment differs

**What Actually Happened:**
 **No failure** - FAISS handles cross-platform serialization correctly

**Why:**
- Modern systems (x86-64 architecture) standardized on little-endian
- FAISS library includes cross-platform compatibility layer
- NumPy/Pickle handle endianness automatically

**If It Had Failed - Recovery Strategy:**
```python
def load(self, path):
    try:
        # Try to load pre-serialized index
        self.index = faiss.read_index(path)
    except Exception as e:
        print(f"Index load failed: {e}")
        print("Rebuilding from embeddings...")
        
        # Fallback: rebuild index from raw embeddings
        embeddings_path = path.replace(".index", "_embeddings.npy")
        embeddings = np.load(embeddings_path).astype('float32')
        
        # Build fresh index on native system
        self.build(embeddings)
        
        # Save optimized binary for next run
        faiss.write_index(self.index, path)
        print("Index rebuilt and cached")
```

**Status:** Not needed; automatic compatibility 

### **ISSUE #7: Streamlit Cloud Resource Constraints**

**Problem:**

Streamlit Cloud limits per app:
- **Memory:** 1 GB
- **CPU:** Single-threaded
- **Disk:** Limited space
- **Session timeout:** 72 hours

Our system requirements:
- FAISS index: 250 MB
- Memory at runtime: ~400 MB (index + embeddings + overhead)
- Execution: <50 ms per query

**Compatibility Check:**
```
FAISS Index:       250 MB   (< 1000 MB available)
Runtime Memory:    400 MB   (< 1000 MB available)
Disk Space Used:   ~600 MB  (models + dependencies)
CPU Usage:         Single   (FAISS is single-threaded)
```

**Result:** Fits within all constraints 

**Future Scaling:**
If dataset grows to 500K+ words:
- Index size could exceed 1 GB
- Solution: Switch to `faiss.IndexIVF` (inverted index for memory-efficient retrieval)
- Or: Use Streamlit Cloud Pro tier (higher resource limits)

## Final Deployment Checklist

Before deploying to production:

-  `.python-version` file exists with `3.11.0`
-  `requirements.txt` updated (no `gensim`, `faiss-cpu>=1.8.0`, `numpy>=1.26.0`)
-  Models tracked with Git LFS (`.gitattributes` committed)
-  All large files pushed via `git push` (LFS automatic)
-  File paths use `os.path.abspath()` (not relative)
-  `load_engine()` decorated with `@st.cache_resource`
-  Local test passes: `streamlit run streamlit_app.py`
-  GitHub logs show LFS pointer files
-  Cloud deployment shows Python 3.11.x

## Deployment Platforms

| Platform | Status | Notes | Link |
|----------|--------|-------|------|
| **Streamlit Cloud** | Production | Python 3.11, LFS supported | [View App](https://phonetic-search-system-23-cp-25.streamlit.app/) |
| **Local (Windows)** | Tested | Python 3.14 works (wheels available) | — |

## How to Debug Issues

### 1. Check Python Version
```bash
python --version  # Should show 3.11.x
cat .python-version
```

### 2. Verify Dependencies Installed
```bash
pip list | grep faiss  # Should show faiss-cpu>=1.8.0
pip list | grep numpy  # Should show numpy>=1.26.0
```

### 3. Verify Models Downloaded
```bash
ls -lh models/
# phonetic_faiss.index  (should be 250 MB, not 50 bytes)
# phonetic_embeddings.pkl (should be 1.3 MB)
# phonetic_words.pkl (should be 1.3 MB)
```

### 4. Test Locally First
```bash
cd phonetic-search-system
streamlit run streamlit_app.py
# Should load in <15 seconds
# Search should complete in <100ms
```

### 5. Check Deployment Logs
- **Streamlit Cloud:** Settings → Manage app → View logs
- **Hugging Face Spaces:** Logs tab at bottom of space page
- Look for:
  ```
  Python 3.11.x environment
  Loading FAISS index...
  App ready at http://...
  ```

### 6. If Models Missing
```bash
# Verify Git LFS setup
git lfs ls-files

# If empty, reinstall LFS:
git lfs install
git lfs pull
```

## 🧪 Testing & Evaluation

### Run Full Evaluation
```bash
cd tests
python evaluation.py
```

This will:
1. Load the search engine
2. Test on 10,000 random words from CMU Dictionary
3. Measure retrieval accuracy @1, @5, @10
4. Calculate average search time
5. Generate `results/evaluation.csv` and charts

### Expected Output
```
Evaluating on 10,000 samples...
Retrieval Accuracy @1:   85.25%
Retrieval Accuracy @5:   99.66%
Retrieval Accuracy @10:  100.00%
Average Search Time:     44.86 ms
Results saved to: results/evaluation.csv
```

## Literature & References

### Phonetic Matching Algorithms
1. **Soundex** - Classic phonetic encoding (1920s)
2. **Metaphone** - Improved phonetic matching
3. **CMU Pronouncing Dictionary** - Comprehensive phoneme database

### Vector Databases
1. **FAISS** - Facebook AI Similarity Search
   - IndexFlatL2: Brute-force L2 distance
   - Used for: Small-to-medium datasets (<10M vectors)
2. **Alternatives:**
   - Pinecone, Weaviate, Milvus for large-scale deployment

### Embeddings
1. **TF-IDF** - Term Frequency-Inverse Document Frequency
   - Used: Character n-grams (2-4 chars)
   - Dimension: 500 features
2. **Alternatives:**
   - Word2Vec, GloVe, FastText for semantic similarity

## License

This project is created as part of Machine Learning & AI course (Semester 6), UET Taxila.

## Author

**Raja Rayyan Ameer**
- GitHub: [@RajaRayyanAmeer](https://github.com/RajaRayyanAmeer)
- Course: Machine Learning & AI
- Semester: 6 (Semester 6 Academic Year 2023-2027)
- Entry: 23-CP-25
- Instructor: Dr. Waqar Ahmad

## Contributing

This is a semester project. For improvements or bug reports, feel free to open an issue.

## Support

If you encounter any issues:

1. **Check the Troubleshooting section** above (covers 90% of issues)
2. **Open an issue on GitHub** with:
   - Python version (`python --version`)
   - Error message
   - Steps to reproduce
3. **Local testing first** before cloud deployment

**Last Updated:** May 31, 2026  
**Status:** Production Ready | Fully Tested | Documented