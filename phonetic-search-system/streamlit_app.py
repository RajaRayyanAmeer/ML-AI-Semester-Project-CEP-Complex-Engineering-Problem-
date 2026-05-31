import streamlit as st
from src.search_engine import PhoneticSearchEngine
from abc import ABC, abstractmethod
import pandas as pd
import time
import os

# INTERFACES (DIP - Depend on abstractions)
class ISearchService(ABC):
    @abstractmethod
    def search(self, query: str, top_k: int):
        pass

class IFormatter(ABC):
    @abstractmethod
    def format_results(self, results):
        pass

# SINGLE RESPONSIBILITY PRINCIPLE
class SearchService(ISearchService):
    """Handles search operations only"""
    def __init__(self, engine: PhoneticSearchEngine):
        self.engine = engine
    
    def search(self, query: str, top_k: int):
        start = time.time()
        results = self.engine.search(query, top_k)
        elapsed = (time.time() - start) * 1000
        return {
            'results': results,
            'elapsed_ms': elapsed,
            'count': len(results)
        }

class ResultFormatter(IFormatter):
    """Formats search results for display"""
    def format_results(self, results):
        df = pd.DataFrame(results)
        df.index = range(1, len(df) + 1)
        df['distance'] = df['distance'].apply(lambda x: f"{x:.4f}")
        return df
    
    def is_exact_match(self, results):
        return results[0]['distance'] < 0.01 if results else False

class StatisticsDisplay:
    """Displays system statistics"""
    def __init__(self, engine: PhoneticSearchEngine):
        self.engine = engine
    
    def render_sidebar(self):
        st.header("📊 System Statistics")
        st.metric("Database Size", f"{len(self.engine.words):,} words")
        st.metric("Accuracy @Top-1", "85.25%")
        st.metric("Accuracy @Top-10", "100%")
        st.metric("Avg Search Time", "44.86 ms")
        
# OPEN/CLOSED PRINCIPLE - Easy to extend
class BaseUIComponent(ABC):
    @abstractmethod
    def render(self):
        pass

class SearchInputComponent(BaseUIComponent):
    """Search input interface"""
    def render(self):
        col1, col2 = st.columns([2, 1])
        with col1:
            query = st.text_input(
                "Enter a word to find phonetic matches:",
                placeholder="e.g., freedom, night, you, thanks",
                help="Type any word to find similar-sounding words"
            )
        with col2:
            top_k = st.slider("Number of results:", 5, 20, 10)
        
        return query, top_k

class SearchResultsComponent(BaseUIComponent):
    """Displays search results"""
    def __init__(self, formatter: ResultFormatter):
        self.formatter = formatter
    
    def render(self, search_response):
        results = search_response['results']
        elapsed = search_response['elapsed_ms']
        
        st.success(f"Found {search_response['count']} matches in {elapsed:.2f}ms")
        
        st.markdown(" Results ")
        df = self.formatter.format_results(results)
        
        st.dataframe(
            df,
            use_container_width=True,
            column_config={
                "word": st.column_config.TextColumn("Word", width="large"),
                "distance": st.column_config.TextColumn("Distance", width="medium")
            }
        )
        
        if self.formatter.is_exact_match(results):
            st.info(f" Exact match found: {results[0]['word']}")

class ExamplesComponent(BaseUIComponent):
    """Example queries section"""
    def render(self):
        st.markdown("---")
        st.markdown(" Try these examples: ")
        
        examples = [
            ("🌙 night", "night"),
            ("🆓 freedom", "freedom"),
            ("👤 you", "you"),
            ("🌊 sea", "sea")
        ]
        
        cols = st.columns(len(examples))
        for col, (label, value) in zip(cols, examples):
            with col:
                if st.button(label, use_container_width=True):
                    st.session_state['example_query'] = value

# MAIN CONTROLLER (Orchestrates components)
class PhoneticSearchApp:
    """Main application controller"""
    def __init__(
        self,
        search_service: ISearchService,
        formatter: IFormatter
    ):
        self.search_service = search_service
        self.formatter = formatter
        self.search_input = SearchInputComponent()
        self.results_display = SearchResultsComponent(formatter)
        self.examples = ExamplesComponent()
    
    def setup_page(self):
        st.set_page_config(
            page_title="Phonetic Search Engine",
            page_icon="🔊",
            layout="wide"
        )
        st.title("🔊 Phonetic Search Engine")
        st.markdown("Find words by sound, not spelling | ML-powered phonetic similarity")
    
    def run(self):
        self.setup_page()
        
        # Get search inputs
        query, top_k = self.search_input.render()
        
        # Handle example queries
        if 'example_query' in st.session_state:
            query = st.session_state['example_query']
            del st.session_state['example_query']
        
        # Search button
        if st.button("🔍 Search", type="primary", use_container_width=True):
            if query:
                response = self.search_service.search(query, top_k)
                self.results_display.render(response)
            else:
                st.warning("Please enter a word to search")
        
        # Examples section
        self.examples.render()
        
        # Footer
        self._render_footer()
    
    def _render_footer(self):
        st.markdown(
            """
            <div style='text-align: center'>
                <p> Phonetic Search Engine | ML & AI Semester Project / CEP </p>
                <p> Raja Rayyan Ameer | 23-CP-25 </p>
                <p> Powered by FastAPI, FAISS, and CMU Pronouncing Dictionary </p>
            </div>
            """,
            unsafe_allow_html=True
        )

# DEPENDENCY INJECTION
@st.cache_resource
def load_engine():
    # Absolute path to the folder streamlit_app.py lives in
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Point directly to the models folder inside that same directory
    model_prefix = os.path.join(base_dir, "models", "phonetic")

    """Load search engine (singleton)"""
    with st.spinner("Loading phonetic search engine..."):
        engine = PhoneticSearchEngine()
        engine.load_all(model_prefix)
    return engine

# APPLICATION ENTRY POINT
def main():
    # Load dependencies
    engine = load_engine()
    
    # Inject dependencies (DIP)
    search_service = SearchService(engine)
    formatter = ResultFormatter()
    stats_display = StatisticsDisplay(engine)
    
    # Render sidebar
    with st.sidebar:
        stats_display.render_sidebar()
    
    # Run main app
    app = PhoneticSearchApp(search_service, formatter)
    app.run()

if __name__ == "__main__":
    main()