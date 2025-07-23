# 🤖 Offline RAG Chatbot

    An advanced, fully offline Retrieval-Augmented Generation (RAG) chatbot that answers queries using only the content from your PDF and Word documents. Designed for privacy, speed, and accuracy—no external APIs required.
## Key Features

   ->100% Offline Operation: All models and processing run locally. No data ever leaves your machine.

   ->Broad Document Support: Ingests and processes both PDF and DOCX files.

   ->Local AI Models: Utilizes powerful, quantized GGUF language models and state-of-the-art sentence transformers.

   ->Optimized for Consumer GPUs: Fine-tuned to run efficiently on a Tesla T4 (16GB VRAM) and other consumer-grade GPUs.

   ->Verifiable Answers: Every response is backed by a source citation, including the original filename and page number.

   ->Intuitive User Interface: A clean and simple UI built with Streamlit, featuring a system status dashboard and a straightforward chat interface.

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit UI  │    │   FastAPI Core   │    │ Document Store  │
│                 │    │                  │    │                 │
│ • File Upload   │◄──►│ • RAG Pipeline   │◄──►│ • Vector DB     │
│ • Chat Interface│    │ • LLM Inference  │    │ • Metadata      │
│ • Status Monitor│    │ • Embedding Gen  │    │ • Chunks        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         v                       v                       v
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│     Frontend    │    │     Backend      │    │    Storage      │
│                 │    │                  │    │                 │
│ • React/HTML    │    │ • sentence-      │    │ • In-memory     │
│ • Tailwind CSS  │    │   transformers   │    │   Vector Store  │
│ • File Handling │    │ • llama-cpp-     │    │ • MongoDB       │
│                 │    │   python         │    │   Metadata      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Technology Stack

➡️ **Application Framework:** Streamlit  
➡️ **LLM Inference:** llama-cpp-python  
➡️ **Embedding Models:** Sentence-Transformers  
➡️ **PDF Processing:** PyMuPDF  
➡️ **DOCX Processing:** python-docx  
➡️ **Language Models (GGUF):**  
➡️ **Default:** TinyLlama-1.1B-Chat-v1.0  
➡️ **Alternative:** Mistral-7B-Instruct-v0.1, Phi-2  

## Quick Start

### Prerequisites

➡️ **Python 3.8+**  
➡️ **GPU:** NVIDIA GPU with CUDA (recommended for best performance) or a CPU  
➡️ **CUDA 11.8+** (for GPU acceleration)  
➡️ **10GB+ free disk space** (for models)  

### Installation

➡️ **Clone and setup environment:**
```bash
cd /app
pip install -r backend/requirements.txt
```

➡️ **Install Dependencies:**  
Create a virtual environment (recommended) and install the required packages.
```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install dependencies
pip install -r backend/requirements.txt
```
> **Note:** The first run will also download the necessary NLTK punkt tokenizer.

➡️ **Run the application:**
```bash
streamlit run app.py
```

## 📖 How to Use

➡️ **Launch the App**  
 Open your web browser and navigate to the local URL provided by Streamlit (usually [http://localhost:8501](http://localhost:8501)).

➡️ **Upload Documents**  
 In the **Document Upload** section, upload one or more PDF or DOCX files.

➡️ **Process Files**  
 For each uploaded file, click the **Process** button. A success message will appear once the file is chunked, vectorized, and stored.

➡️ **Ask Questions**  
 Once at least one document is processed, use the **Chat Interface** to ask questions about the content of your documents.

## Models Used

### Embedding Model

➡️ **Model:** `sentence-transformers/all-MiniLM-L6-v2`  
➡️ **Dimensions:** 384  
➡️ **Size:** ~90MB  
➡️ **Performance:** Fast, good quality embeddings

### LLM Options (Quantized GGUF)

➡️ **Primary:** TinyLlama-1.1B Q4_K_M (~0.6GB)  
➡️ **Alternative:** Phi-2 Q4_K_M (~1.6GB) ,Mistral-7B-Instruct Q4_K_M (~4.1GB) 


## Document Processing Pipeline

### 1. Text Extraction

➡️ **PDF:** PyMuPDF (`fitz`) for accurate text extraction  
➡️ **DOCX:** `python-docx` for structured document parsing  
➡️ **Metadata:** Captures filename and page numbers

### 2. Chunking Strategy

```
Semantic Sentence-Based Chunking with Sliding Window Overlap

┌─────────────────────────────────────────────────────────┐
│  Document Page 1                                        │
│  ┌─────────────────┐  ┌─────────────────┐              │
│  │    Chunk 1      │  │    Chunk 2      │              │
│  │   (200 words)   │  │   (200 words)   │              │
│  │                 │  │                 │              │
│  └─────────────────┘  └─────────────────┘              │
│        │                     │                         │
│        └─────────────────────┼─── 30 word overlap      │
│                              │                         │
└─────────────────────────────────────────────────────────┘

Benefits:
• Maintains semantic coherence
• Prevents information loss at boundaries  
• Enables precise retrieval
• Supports accurate citation
```

**Why This Strategy?**
- **Semantic Coherence**: Sentence boundaries prevent cut-off context
- **Overlap Window**: 30-token overlap ensures no information loss
- **Traceability**: Each chunk tagged with page number and document origin
- **Retrieval Precision**: Focused chunks improve relevance matching

### 3. Metadata Storage
Each chunk stores:
```json
{
  "chunk_id": "uuid-string",
  "filename": "document.pdf", 
  "page_number": 3,
  "text_content": "The extracted text content...",
  "embedding": [0.1, -0.2, 0.8, ...]
}
```

## 💻 Hardware Optimization

- This application is optimized to run on a Tesla T4 GPU (16GB VRAM), but it is also compatible with a wide range of consumer GPUs and can run on a CPU if needed.
- To ensure performance and stay within the 16GB VRAM limit, the following strategies have been implemented in `backend/model_loader.py`:
   - **Lightweight Default Model**: TinyLlama-1.1B is used by default, offering a great balance of performance and a very low memory footprint (~0.6 GB).
   - **Full GPU Offloading**: By setting `n_gpu_layers: -1`, all model layers are offloaded to the GPU, maximizing inference speed.
   - **Optimized Context Window**: The context size (`n_ctx`) is set to 2048 tokens, which is sufficient for RAG tasks while conserving VRAM.
   - **Efficient Data Types**: `f16_kv = True` is enabled to use half-precision for the key/value cache in memory, further reducing VRAM usage.
- **Tesla T4 GPU Configuration Example**:
   ```python
   # Optimized for 16GB VRAM
   model_config = {
         "n_gpu_layers": -1,      # Offload all layers to GPU
         "n_ctx": 4096,           # Context window  
         "n_batch": 512,          # Batch size
         "memory_f16": True       # Use half precision
   }
   ```
- **Performance Targets**:
   - Response Time: ≤ 15 seconds
   - Memory Usage: < 12GB VRAM
   - Throughput: 25-35 tokens/second
   - Embedding Speed: ~500 chunks/second
   This application is optimized to run on a Tesla T4 GPU (16GB VRAM), but it is also compatible with a wide range of consumer GPUs and can run on a CPU if needed.

To ensure performance and stay within the 16GB VRAM limit, the following strategies have been implemented in backend/model_loader.py

Lightweight Default Model: TinyLlama-1.1B is used by default, offering a great balance of performance and a very low memory footprint (~0.6 GB).

Full GPU Offloading: By setting n_gpu_layers: -1, all model layers are offloaded to the GPU, maximizing inference speed.

Optimized Context Window: The context size (n_ctx) is set to 2048 tokens, which is sufficient for RAG tasks while conserving VRAM.

Efficient Data Types: f16_kv = True is enabled to use half-precision for the key/value cache in memory, further reducing VRAM usage.

### Tesla T4 GPU Configuration
```python
# Optimized for 16GB VRAM
model_config = {
    "n_gpu_layers": -1,      # Offload all layers to GPU
    "n_ctx": 4096,          # Context window  
    "n_batch": 512,         # Batch size
    "memory_f16": True      # Use half precision
}
```

### Performance Targets
- **Response Time**: ≤ 15 seconds
- **Memory Usage**: < 12GB VRAM
- **Throughput**: 25-35 tokens/second
- **Embedding Speed**: ~500 chunks/second

## File Structure

```
/app/
├── app.py                 # Streamlit main application
├── backend/
│   ├── rag_pipeline.py    # Core RAG logic
│   ├── model_loader.py    # LLM loading and inference
│   ├── server.py          # FastAPI endpoints (optional)
│   └── requirements.txt   # Python dependencies
├── models/                # Downloaded model cache
│   ├── all-MiniLM-L6-v2/ # Embedding model
│   └── *.gguf            # Quantized LLM models
├── docs/                 # Sample documents and screenshots
└── README.md             # This file
```

### Performance Validation
```bash
# Test document processing
python -c "
from backend.rag_pipeline import RAGPipeline
rag = RAGPipeline()
result = rag.process_and_store_document('sample.pdf')
print(result)
"

# Test query performance  
python -c "
from backend.model_loader import RAGWithLLM
system = RAGWithLLM()
system.initialize()
response = system.answer_question('What is this document about?')
print(response)
"
```

## Configuration

### Environment Variables
```bash
# GPU Settings (optional)
CUDA_VISIBLE_DEVICES=0
TOKENIZERS_PARALLELISM=false

# Model Settings
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
LLM_MODEL=TinyLlama-1.1B-Chat-v1.0
MAX_CHUNK_SIZE=200
OVERLAP_SIZE=30
```


## Monitoring

The application provides real-time monitoring of:

- **GPU Usage**: Memory utilization and temperature
- **Processing Stats**: Documents processed, chunks created
- **Response Times**: Query processing and generation times
- **Model Status**: Loading state and configuration
- **System Health**: Memory usage and performance metrics

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce context size or use smaller model
   n_ctx=2048, n_batch=256
   ```

2. **Slow Embedding Generation**  
   ```bash
   # Check GPU utilization
   nvidia-smi -l 1
   
   # Ensure model is on GPU
   model.encode(texts, device='cuda')
   ```

3. **Model Download Fails**
   ```bash
   # Manual download
   wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf
   mv *.gguf models/
   ```

4. **PDF Extraction Issues**
   ```python
   # For complex PDFs, try OCR
   pip install pytesseract
   # Configure in rag_pipeline.py
   ```

## Security & Privacy

- **No External Calls**: All processing happens locally
- **Data Isolation**: Documents never leave your system
- **Model Caching**: All AI models stored locally
- **No Logging**: No sensitive data logged externally

## Production Deployment

For production use, consider:

1. **Model Pre-caching**: Download all models during build
2. **Resource Limits**: Configure memory and GPU limits
3. **Monitoring**: Add comprehensive logging and metrics
4. **Backup**: Regular backup of processed documents
5. **Security**: Add authentication and rate limiting


**🎯 Ready to get started? Upload your first document and ask a question!**