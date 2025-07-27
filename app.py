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
except ImportError as e:
    st.error(f"Failed to import RAG components: {e}")
    st.stop()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Offline RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for a Modern, Professional UI ---
st.markdown("""
<style>
    /* --- Base & Theme --- */
    .stApp {
        background-color: #0d1117; /* GitHub dark background */
        color: #c9d1d9; /* GitHub dark text */
        font-size: 14px; /* Adjusted base font size */
    }
    h1, h2, h3, h4, h5, h6, p, label, .st-emotion-cache-16idsys p {
        color: #c9d1d9;
    }
    
    /* --- Sidebar --- */
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    [data-testid="stSidebar"] .st-emotion-cache-1cypcdb { /* Sidebar headers */
        color: #c9d1d9;
        font-weight: 600;
    }

    /* --- Main Content --- */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 2rem;
    }

    /* --- Chat Bubbles --- */
    .user-bubble, .assistant-bubble {
        padding: 12px 18px;
        border-radius: 18px;
        margin-bottom: 10px;
        max-width: 80%;
        clear: both;
        line-height: 1.6;
    }
    .user-bubble {
        background-color: #21262d;
        float: right;
    }
    .assistant-bubble {
        background-color: #161b22;
        border: 1px solid #30363d;
        float: left;
    }

    /* --- Highlighted Source Info --- */
    .source-info-text {
        font-size: 0.9rem;
        color: #8b949e;
        margin-top: 12px;
        padding-top: 12px;
        border-top: 1px solid #30363d;
    }
    .source-info-text strong {
        color: #58a6ff; /* Highlight color for parameter values */
        font-weight: 600;
    }

    /* --- Interactive Elements --- */
    .stButton>button {
        border: 1px solid #30363d;
        background-color: #21262d;
        color: #58a6ff;
        padding: 10px 16px;
        transition: all 0.2s ease-in-out;
        font-weight: 600;
        border-radius: 6px;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #30363d;
        border-color: #8b949e;
    }
    
    /* --- File Uploader --- */
    [data-testid="stFileUploader"] {
        background-color: #161b22;
        border: 2px dashed #30363d;
        border-radius: 8px;
        padding: 1rem;
    }
    
    .status-pill {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 500;
        text-align: center;
    }
    .status-processed { background-color: #238636; color: white; }
    .status-error { background-color: #da3633; color: white; }

</style>
""", unsafe_allow_html=True)

# --- Function Definitions ---
@st.cache_resource
def initialize_rag_system():
    """Initialize the RAG system (cached for performance)"""
    try:
        rag_system = RAGWithLLM()
        init_result = rag_system.initialize(model_name="tinyllama")
        return rag_system, init_result
    except Exception as e:
        st.error(f"Failed to initialize RAG system: {e}")
        logger.error(f"Initialization failed: {e}", exc_info=True)
        return None, None

# --- Initialize Session State ---
if "rag_system" not in st.session_state:
    with st.spinner("Initializing RAG System... This may take a moment."):
        st.session_state.rag_system, st.session_state.init_result = initialize_rag_system()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "uploaded_files_status" not in st.session_state:
    st.session_state.uploaded_files_status = {}

# --- Sidebar ---
with st.sidebar:
    st.header("System Status")
    init_result = st.session_state.get("init_result", {})
    if init_result:
        gpu_info = init_result.get("gpu_info", {})
        if gpu_info.get("available"):
            st.write("✅ GPU Available")
        else:
            st.write("⚠️ No GPU detected")
        
        llm_info = st.session_state.rag_system.llm_loader.get_model_info()
        if llm_info.get("loaded"):
            st.write(f"✅ LLM Loaded: `{llm_info.get('model_name', 'N/A')}`")
        else:
            st.write("❌ LLM Model Failed")
        
        if init_result.get("system_ready"):
            st.write("✅ System Ready")
        else:
            st.write("❌ System Not Ready")
            
    st.markdown("---")
    st.header("Stats")
    try:
        status = st.session_state.rag_system.get_system_status()
        rag_stats = status.get("rag_stats", {})
        st.write(f"**Documents Processed:** {rag_stats.get('documents_processed', 0)}")
        st.write(f"**Total Chunks:** {rag_stats.get('vector_store_stats', {}).get('total_chunks', 0)}")
    except Exception:
        st.write("**Documents Processed:** 0")
        st.write("**Total Chunks:** 0")

# --- Main Content ---
st.markdown('<h1 class="main-header">Offline RAG Chatbot</h1>', unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("Document Upload")
    uploaded_files = st.file_uploader(
        "Drag and drop files here",
        accept_multiple_files=True,
        type=['pdf', 'docx'],
        label_visibility="collapsed"
    )
    
    if uploaded_files:
        for f in uploaded_files:
            if f.name not in st.session_state.uploaded_files_status:
                st.session_state.uploaded_files_status[f.name] = {"status": "pending", "file": f}

    st.subheader("Uploaded Files")
    if not st.session_state.uploaded_files_status:
        st.info("No files uploaded yet.")
    else:
        for name, data in st.session_state.uploaded_files_status.items():
            file_col, status_col = st.columns([4, 1])
            with file_col:
                st.write(f"📄 {name}")
            with status_col:
                if data["status"] == "processed":
                    st.markdown('<span class="status-pill status-processed">Processed</span>', unsafe_allow_html=True)
                elif data["status"] == "error":
                    st.markdown('<span class="status-pill status-error">Error</span>', unsafe_allow_html=True)
                else:
                    if st.button("Process", key=f"process_{name}"):
                        with st.spinner(f"Processing {name}..."):
                            try:
                                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(name).suffix) as tmp_file:
                                    tmp_file.write(data["file"].getvalue())
                                    tmp_path = tmp_file.name
                                result = st.session_state.rag_system.process_document(tmp_path)
                                os.unlink(tmp_path)
                                if result.get("status") == "success":
                                    st.session_state.uploaded_files_status[name]["status"] = "processed"
                                else:
                                    st.session_state.uploaded_files_status[name]["status"] = "error"
                                st.rerun()
                            except Exception as e:
                                st.session_state.uploaded_files_status[name]["status"] = "error"
                                logger.error(f"Error processing {name}: {e}")
                                st.rerun()

with col2:
    st.subheader("Chat Interface")
    
    chat_container = st.container(height=500)
    with chat_container:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"<div class='user-bubble'>{message['content']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='assistant-bubble'>{message['content']}", unsafe_allow_html=True)
                if message.get("source"):
                    source = message["source"]
                    metadata = message.get("llm_metadata", {})
                    gen_time = metadata.get("generation_time", "N/A")
                    
                    st.markdown(f"""
                    <div class="source-info-text">
                        Source: <strong>{source["filename"]} (Page: {source["page_number"]})</strong> | 
                        Similarity: <strong>{message.get("similarity_score", 0.0):.2f}</strong> | 
                        Time: <strong>{gen_time}</strong>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)


    user_question = st.chat_input("Ask a question about your documents...")
    if user_question:
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        with st.spinner("Searching..."):
            result = st.session_state.rag_system.answer_question(user_question)
            
            chat_message = {
                "role": "assistant",
                "content": result["answer"],
                "source": result.get("source"),
                "similarity_score": result.get("similarity_score"),
                "llm_metadata": result.get("llm_metadata")
            }
            st.session_state.chat_history.append(chat_message)
            st.rerun()
#commited