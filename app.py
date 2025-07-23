# Offline RAG Chatbot - Streamlit Interface
import streamlit as st
import os
import sys
import tempfile
import logging
from pathlib import Path
import time

# Add backend to path
sys.path.append(str(Path(__file__).resolve().parent / "backend"))



# Import our RAG components
try:
    from model_loader import RAGWithLLM
    from rag_pipeline import RAGPipeline

except ImportError as e:
    st.error(f"Failed to import RAG components: {e}")
    st.stop()

# Configure logging
logging.basicConfig(level=logging.WARNING)  # Reduced logging for cleaner output
logger = logging.getLogger(__name__)

# Streamlit page config
st.set_page_config(
    page_title="Offline RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stats-box {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        color: #333;
    }
    .source-info {
        background: #e1f5fe;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
        color: #01579b; /* Darker blue text for better contrast */
    }
    .source-info h4 {
        color: #01579b;
    }
    .error-box {
        background: #ffebee;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #f44336;
        margin: 1rem 0;
        color: #c62828; /* Darker red text */
    }
    .error-box h4 {
        color: #c62828;
    }
    .success-box {
        background: #e8f5e9;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4caf50;
        margin: 1rem 0;
        color: #1b5e20; /* Darker green text */
    }
    .success-box h4 {
        color: #1b5e20;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_rag_system():
    """Initialize the RAG system (cached for performance)"""
    try:
        rag_system = RAGWithLLM()
        # Initialize with the lightweight 'tinyllama' model by default
        init_result = rag_system.initialize(model_name="tinyllama")
        return rag_system, init_result
    except Exception as e:
        st.error(f"Failed to initialize RAG system: {e}")
        logger.error(f"Initialization failed: {e}", exc_info=True)
        return None, None

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Offline RAG Chatbot</h1>
        <p>Ask questions about your uploaded PDF and DOCX documents</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize system
    rag_system, init_result = initialize_rag_system()
    
    if rag_system is None:
        st.error("❌ Failed to initialize RAG system. Please check the logs for more details.")
        return
    
    # Sidebar - System Information
    with st.sidebar:
        st.header("📊 System Status")
        
        if init_result:
            # GPU Information
            gpu_info = init_result.get("gpu_info", {})
            if gpu_info.get("available"):
                st.success("🟢 GPU Available")
                for gpu in gpu_info.get("gpus", []):
                    st.write(f"**{gpu['name']}**")
                    st.write(f"Memory: {gpu['memory_free_mb']}MB free / {gpu['memory_total_mb']}MB total")
            else:
                st.warning("🟡 No GPU detected - using CPU")
            
            # Model Status
            llm_info = rag_system.llm_loader.get_model_info()
            if llm_info.get("loaded"):
                st.success(f"🟢 {llm_info.get('model_name', 'LLM')} Loaded")
            else:
                st.warning("🟡 LLM Model Loading Failed")
            
            # System Ready
            if init_result.get("system_ready"):
                st.success("✅ System Ready")
            else:
                st.error("❌ System Not Ready")
        
        # Get current system stats
        if st.button("🔄 Refresh Status"):
            st.rerun()
        
        try:
            status = rag_system.get_system_status()
            rag_stats = status.get("rag_stats", {})
            
            st.markdown("### 📈 Statistics")
            st.write(f"**Documents Processed:** {rag_stats.get('documents_processed', 0)}")
            st.write(f"**Total Chunks:** {rag_stats.get('vector_store_stats', {}).get('total_chunks', 0)}")
            st.write(f"**Embedding Model:** `{rag_stats.get('embedding_model', 'N/A').split('/')[-1]}`")
            
        except Exception as e:
            st.error(f"Failed to get system status: {e}")

    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📄 Document Upload")
        
        # File upload
        uploaded_files = st.file_uploader(
            "Upload PDF or DOCX files",
            accept_multiple_files=True,
            type=['pdf', 'docx'],
            help="Upload one or more PDF or DOCX documents to add to the knowledge base"
        )
        
        # Process uploaded files
        if uploaded_files:
            for uploaded_file in uploaded_files:
                # Use a unique key for each button based on file name and size
                button_key = f"process_{uploaded_file.name}_{uploaded_file.size}"
                if st.button(f"Process {uploaded_file.name}", key=button_key):
                    with st.spinner(f"Processing {uploaded_file.name}..."):
                        try:
                            # Save uploaded file temporarily
                            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                                tmp_file.write(uploaded_file.getvalue())
                                tmp_path = tmp_file.name
                            
                            # Process the document
                            result = rag_system.process_document(tmp_path)
                            
                            # Clean up temp file
                            os.unlink(tmp_path)
                            
                            # Show results
                            if result.get("status") == "success":
                                st.markdown(f"""
                                <div class="success-box">
                                    <h4>✅ {result['filename']} processed!</h4>
                                    <p><strong>Chunks created:</strong> {result['chunks_created']}</p>
                                    <p><strong>Processing time:</strong> {result['processing_time']}</p>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                <div class="error-box">
                                    <h4>❌ Error processing {result['filename']}</h4>
                                    <p>{result['error']}</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                        except Exception as e:
                            st.markdown(f"""
                            <div class="error-box">
                                <h4>❌ Error processing {uploaded_file.name}</h4>
                                <p>{str(e)}</p>
                            </div>
                            """, unsafe_allow_html=True)

    with col2:
        st.header("💬 Chat Interface")
        
        # Check if documents are available
        try:
            status = rag_system.get_system_status()
            total_chunks = status.get("rag_stats", {}).get("vector_store_stats", {}).get('total_chunks', 0)
            
            if total_chunks == 0:
                st.warning("⚠️ No documents processed yet. Please upload and process a file.")
            else:
                st.info(f"📚 Ready to answer questions from {total_chunks} document chunks.")
        except Exception as e:
            st.error(f"Failed to check document status: {e}")
            total_chunks = 0
        
        # Chat interface
        if total_chunks > 0:
            # Question input
            user_question = st.text_area(
                "Ask a question about your documents:",
                height=100,
                placeholder="e.g., What is the main topic discussed in the document?"
            )
            
            # Submit button
            if st.button("🔍 Ask Question", disabled=not user_question.strip()):
                with st.spinner("Searching documents and generating answer..."):
                    try:
                        start_time = time.time()
                        result = rag_system.answer_question(user_question)
                        total_time = time.time() - start_time
                        
                        # Display answer
                        st.markdown("### 💡 Answer")
                        st.write(result["answer"])
                        
                        # Display source information
                        if result.get("source"):
                            source = result["source"]
                            similarity = result.get("similarity_score", 0.0)
                            
                            st.markdown(f"""
                            <div class="source-info">
                                <h4>📖 Source Information</h4>
                                <p><strong>File:</strong> {source["filename"]}</p>
                                <p><strong>Page:</strong> {source["page_number"]}</p>
                                <p><strong>Similarity:</strong> {similarity:.2f}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                             st.markdown("""
                            <div class="error-box">
                                <h4>ℹ️ No relevant information found</h4>
                                <p>The question could not be answered based on the uploaded documents.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Show LLM metadata if available
                        if result.get("llm_metadata"):
                            metadata = result["llm_metadata"]
                            st.markdown("### 🔧 Generation Details")
                            gen_time = metadata.get("generation_time", "0.0s")
                            st.write(f"**Total Time:** {total_time:.2f}s (Generation: {gen_time})")
                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                st.metric("Prompt Tokens", metadata.get("prompt_tokens", 0))
                            with col_b:
                                st.metric("Completion Tokens", metadata.get("completion_tokens", 0))
                            with col_c:
                                st.metric("Total Tokens", metadata.get("total_tokens", 0))
                        
                    except Exception as e:
                        st.markdown(f"""
                        <div class="error-box">
                            <h4>❌ Error processing question</h4>
                            <p>{str(e)}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        logger.error(f"Error during question answering: {e}", exc_info=True)
        
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>🤖 Offline RAG Chatbot | Powered by sentence-transformers & quantized LLMs</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
