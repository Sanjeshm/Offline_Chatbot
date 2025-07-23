# Core RAG pipeline for offline document-based chatbot
import os
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import nltk
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF for PDF processing
import docx  # python-docx for DOCX processing
from pathlib import Path
import uuid
import time
import numpy as np
import torch

logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    chunk_id: str
    filename: str
    page_number: int
    text_content: str
    embedding: Optional[np.ndarray] = None

class DocumentProcessor:
    """Handles PDF and DOCX document processing with metadata extraction"""
    
    def __init__(self):
        # Download required NLTK data for sentence tokenization
        try:
            nltk.data.find('tokenizers/punkt')
        except nltk.downloader.DownloadError:
            nltk.download('punkt', quiet=True)

    def extract_text_from_pdf(self, file_path: str) -> List[Dict]:
        """Extract text from PDF with page numbers"""
        pages = []
        try:
            doc = fitz.open(file_path)
            for page_num, page in enumerate(doc):
                text = page.get_text("text")
                if text.strip():
                    pages.append({
                        'text': text.strip(),
                        'page_number': page_num + 1
                    })
            doc.close()
        except Exception as e:
            logger.error(f"Error extracting PDF {file_path}: {e}")
        return pages

    def extract_text_from_docx(self, file_path: str) -> List[Dict]:
        """Extract text from DOCX with page approximation"""
        pages = []
        try:
            doc = docx.Document(file_path)
            full_text = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
            
            all_text = '\n'.join(full_text)
            # Simple page approximation
            words = all_text.split()
            words_per_page = 400
            
            for i in range(0, len(words), words_per_page):
                page_text = ' '.join(words[i:i + words_per_page])
                if page_text:
                    pages.append({
                        'text': page_text,
                        'page_number': (i // words_per_page) + 1
                    })
        except Exception as e:
            logger.error(f"Error extracting DOCX {file_path}: {e}")
        return pages

    def semantic_chunk_text(self, pages: List[Dict], filename: str, chunk_size: int = 256, overlap: int = 50) -> List[DocumentChunk]:
        """Create semantic chunks with sentence boundaries and overlap"""
        chunks = []
        
        for page in pages:
            try:
                sentences = nltk.sent_tokenize(page['text'])
            except Exception:
                sentences = page['text'].replace('\n', ' ').split('. ')

            current_chunk_tokens = []
            for sentence in sentences:
                sentence_tokens = sentence.split()
                if len(current_chunk_tokens) + len(sentence_tokens) <= chunk_size:
                    current_chunk_tokens.extend(sentence_tokens)
                else:
                    if current_chunk_tokens:
                        chunks.append(DocumentChunk(
                            chunk_id=str(uuid.uuid4()),
                            filename=filename,
                            page_number=page['page_number'],
                            text_content=' '.join(current_chunk_tokens)
                        ))
                    # Start new chunk with overlap
                    overlap_tokens = current_chunk_tokens[-overlap:]
                    current_chunk_tokens = overlap_tokens + sentence_tokens
            
            if current_chunk_tokens:
                chunks.append(DocumentChunk(
                    chunk_id=str(uuid.uuid4()),
                    filename=filename,
                    page_number=page['page_number'],
                    text_content=' '.join(current_chunk_tokens)
                ))
        
        return chunks

    def process_document(self, file_path: str) -> List[DocumentChunk]:
        """Process a document (PDF or DOCX) into chunks"""
        filename = os.path.basename(file_path)
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.pdf':
            pages = self.extract_text_from_pdf(file_path)
        elif file_ext == '.docx':
            pages = self.extract_text_from_docx(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        if not pages:
            raise ValueError(f"No text extracted from {filename}")
        
        chunks = self.semantic_chunk_text(pages, filename)
        logger.info(f"Processed {filename}: {len(pages)} pages → {len(chunks)} chunks")
        return chunks


class EmbeddingEngine:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the embedding model (downloads if not cached)"""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            # Use CUDA if available, otherwise CPU
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model = SentenceTransformer(self.model_name, device=device)
            logger.info(f"Embedding model loaded successfully on '{device}'. Vector dim: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            # Fallback to CPU if CUDA fails
            try:
                self.model = SentenceTransformer(self.model_name, device='cpu')
                logger.info("Successfully loaded embedding model on CPU as a fallback.")
            except Exception as e_cpu:
                logger.error(f"CPU fallback for embedding model also failed: {e_cpu}")
                raise

    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding vectors"""
        return self.model.get_sentence_embedding_dimension()

    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        return self.model.encode(text, convert_to_numpy=True)

    def embed_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Generate embeddings for all chunks"""
        logger.info(f"Generating embeddings for {len(chunks)} chunks...")
        texts = [chunk.text_content for chunk in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
        
        return chunks


class VectorStore:
    """Simple in-memory vector store with cosine similarity search"""
    
    def __init__(self):
        self.chunks: List[DocumentChunk] = []
        self._embeddings_matrix: Optional[np.ndarray] = None

    def add_chunks(self, chunks: List[DocumentChunk]):
        """Add chunks to the vector store"""
        new_embeddings = [chunk.embedding for chunk in chunks if chunk.embedding is not None]
        if not new_embeddings:
            logger.warning("No chunks with embeddings provided to add.")
            return

        new_embeddings_np = np.array(new_embeddings, dtype=np.float32)

        if self._embeddings_matrix is None:
            self._embeddings_matrix = new_embeddings_np
        else:
            self._embeddings_matrix = np.vstack([self._embeddings_matrix, new_embeddings_np])
        
        self.chunks.extend(chunks)
        logger.info(f"Added {len(chunks)} chunks. Total in store: {len(self.chunks)}")

    def _cosine_similarity_matrix(self, query_embedding: np.ndarray) -> np.ndarray:
        """Calculate cosine similarity using matrix operations for speed."""
        query_norm = np.linalg.norm(query_embedding)
        matrix_norm = np.linalg.norm(self._embeddings_matrix, axis=1)
        
        # Dot product
        dot_product = np.dot(self._embeddings_matrix, query_embedding)
        
        # Cosine similarity
        similarities = dot_product / (matrix_norm * query_norm)
        return similarities

    def search(self, query_embedding: np.ndarray, top_k: int = 1) -> List[Tuple[DocumentChunk, float]]:
        """Search for most similar chunks (returns exactly top_k results)"""
        if self._embeddings_matrix is None or len(self.chunks) == 0:
            return []
        
        # Normalize query embedding
        query_embedding_norm = query_embedding / np.linalg.norm(query_embedding)

        # Normalize the embeddings matrix
        embeddings_matrix_norm = self._embeddings_matrix / np.linalg.norm(self._embeddings_matrix, axis=1, keepdims=True)

        # Compute cosine similarities
        similarities = np.dot(embeddings_matrix_norm, query_embedding_norm)
        
        # Get top_k indices. Using argpartition for efficiency.
        # It finds the k-th largest value's index without sorting the whole array.
        # We get the indices of the top_k largest values.
        top_k_indices = np.argpartition(similarities, -top_k)[-top_k:]
        
        # Sort only the top_k results
        results = sorted([(self.chunks[i], similarities[i]) for i in top_k_indices], key=lambda x: x[1], reverse=True)
        
        return results

    def get_stats(self) -> Dict:
        """Get vector store statistics"""
        return {
            "total_chunks": len(self.chunks),
            "unique_documents": len(set(chunk.filename for chunk in self.chunks)),
            "dimension": self._embeddings_matrix.shape[1] if self._embeddings_matrix is not None else 0
        }


class RAGPipeline:
    """Main RAG pipeline coordinator"""
    
    def __init__(self):
        self.doc_processor = DocumentProcessor()
        self.embedding_engine = EmbeddingEngine()
        self.vector_store = VectorStore()
        self.documents_processed = 0

    def process_and_store_document(self, file_path: str) -> Dict:
        """Process a document and store it in the vector store"""
        start_time = time.time()
        
        try:
            chunks = self.doc_processor.process_document(file_path)
            chunks_with_embeddings = self.embedding_engine.embed_chunks(chunks)
            self.vector_store.add_chunks(chunks_with_embeddings)
            self.documents_processed += 1
            processing_time = time.time() - start_time
            
            return {
                "filename": os.path.basename(file_path),
                "chunks_created": len(chunks),
                "processing_time": f"{processing_time:.2f}s",
                "total_documents": self.documents_processed,
                "status": "success"
            }
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")
            return {
                "filename": os.path.basename(file_path),
                "error": str(e),
                "status": "error"
            }


    def query(self, question: str) -> Dict:
        """Execute RAG query: retrieve most relevant chunk"""
        start_time = time.time()
        
        query_embedding = self.embedding_engine.embed_text(question)
        search_results = self.vector_store.search(query_embedding, top_k=1)
        
        processing_time = time.time() - start_time

        if not search_results:
            return {
                "question": question,
                "retrieved_chunk": None,
                "source": None,
                "similarity_score": 0.0,
                "processing_time": f"{processing_time:.2f}s"
            }
        
        best_chunk, similarity_score = search_results[0]
        
        return {
            "question": question,
            "retrieved_chunk": best_chunk.text_content,
            "source": {
                "filename": best_chunk.filename,
                "page_number": best_chunk.page_number,
                "chunk_id": best_chunk.chunk_id
            },
            "similarity_score": float(similarity_score),
            "processing_time": f"{processing_time:.2f}s"
        }

    def get_system_stats(self) -> Dict:
        """Get comprehensive system statistics"""
        return {
            "documents_processed": self.documents_processed,
            "vector_store_stats": self.vector_store.get_stats(),
            "embedding_model": self.embedding_engine.model_name,
            "embedding_dimension": self.embedding_engine.get_embedding_dimension()
        }
