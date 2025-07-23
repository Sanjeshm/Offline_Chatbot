# Model loader for offline LLM inference using quantized GGUF models
import os
import logging
from typing import Optional, Dict
from pathlib import Path
import requests
import time
from llama_cpp import Llama

os.environ['LLAMA_SET_ROWS'] = '1'


logger = logging.getLogger(__name__)

class LLMLoader:
    """Handles loading and managing quantized GGUF models for offline inference"""
    
    def __init__(self):
        self.model = None
        self.model_path = None
        self.models_dir = Path(__file__).resolve().parent / "models"
        self.models_dir.mkdir(exist_ok=True)
        
        # Model download URLs (these would be downloaded once and cached)
        self.model_urls = {
            "mistral-7b-instruct": {
                "filename": "mistral-7b-instruct-v0.1.Q4_K_M.gguf",
                "url": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
                "size": "4.1GB"
            },
            "phi-2": {
                "filename": "phi-2.Q4_K_M.gguf", 
                "url": "https://huggingface.co/microsoft/phi-2/resolve/main/phi-2.Q4_K_M.gguf",
                "size": "1.6GB"
            },
            "tinyllama": {
                "filename": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
                "url": "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
                "size": "0.6GB"
            }
        }
    def check_gpu_availability(self) -> Dict:
        """Check GPU availability and memory"""
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.free', 
                                   '--format=csv,noheader,nounits'], 
                                   capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                gpu_info = []
                for line in lines:
                    name, total, free = line.split(', ')
                    gpu_info.append({
                        "name": name.strip(),
                        "memory_total_mb": int(total),
                        "memory_free_mb": int(free)
                    })
                return {"available": True, "gpus": gpu_info}
            else:
                return {"available": False, "error": "nvidia-smi not found"}
        except Exception as e:
            return {"available": False, "error": str(e)}

    def download_model(self, model_name: str) -> bool:
        """Download a model if it doesn't exist locally"""
        if model_name not in self.model_urls:
            logger.error(f"Unknown model: {model_name}")
            return False
        
        model_info = self.model_urls[model_name]
        model_path = self.models_dir / model_info["filename"]
        
        if model_path.exists():
            logger.info(f"Model {model_name} already exists at {model_path}")
            self.model_path = model_path
            return True
        
        logger.info(f"Downloading {model_name} ({model_info['size']}) to {model_path}")
        
        try:
            with requests.get(model_info["url"], stream=True) as r:
                r.raise_for_status()
                with open(model_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            logger.info(f"Model {model_name} downloaded successfully.")
            self.model_path = model_path
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download model {model_name}: {e}")
            return False


    def load_model(self, model_name: str = "tinyllama") -> bool:
        """Load a quantized GGUF model for inference"""
        try:
            if not self.download_model(model_name):
                # If download fails, check if the file exists anyway (e.g., manual download)
                model_info = self.model_urls.get(model_name)
                if not model_info or not (self.models_dir / model_info["filename"]).exists():
                    logger.error(f"Model '{model_name}' is not available.")
                    return False
            
            model_path = self.model_path # path is set by download_model

            logger.info(f"Loading model: {model_name} from {model_path}")

            # Offload all layers to GPU if possible, suitable for smaller models on low-VRAM GPUs
            self.model_config = {
                "model_path": str(model_path),
                "n_gpu_layers": -1, 
                "n_ctx": 2048,      # Reduced context window for lower memory usage
                "n_batch": 512,     
                "verbose": False,
                "f16_kv": True,
                "use_mlock": False, # Set to False on systems with limited memory
                "use_mmap": True
            }

            self.model = Llama(
                model_path=str(model_path),
                n_ctx=self.model_config["n_ctx"],
                n_batch=self.model_config["n_batch"],
                n_gpu_layers=self.model_config["n_gpu_layers"],
                verbose=self.model_config["verbose"],
                f16_kv=self.model_config["f16_kv"],                
                use_mlock=self.model_config["use_mlock"], 
                use_mmap=self.model_config["use_mmap"]
            )

            self.model_name = model_path.stem 
            logger.info(f"Model {model_name} loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return False

    def generate_response(self, prompt: str, max_tokens: int = 256) -> Dict:
        """Generate response using the loaded model"""
        if self.model is None:
            return {
                "error": "No model loaded. Call load_model() first.",
                "response": None,
                "metadata": None
            }
        
        start_time = time.time()
        
        try:
            output = self.model(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=0.2,
                top_p=0.9,
                stop=["</s>", "[INST]", "\n\nHuman:", "\n\nUser:", "<|im_end|>"]
            )
            response_text = output["choices"][0]["text"]
            generation_time = time.time() - start_time

            return {
                 "response": response_text.strip(),
                 "metadata": {
                    "model": self.model_path.name,
                    "prompt_tokens": output.get("usage", {}).get("prompt_tokens", 0),
                    "completion_tokens": output.get("usage", {}).get("completion_tokens", 0),
                    "total_tokens": output.get("usage", {}).get("total_tokens", 0),
                    "generation_time": f"{generation_time:.2f}s"
                 },
                "error": None
            }
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return {
                "error": str(e),
                "response": None,
                "metadata": None
            }

    def _format_rag_prompt(self, user_query: str, context: str = None) -> str:
        """Format the prompt for RAG with retrieved context"""
        # Using TinyLlama chat template
        if context:
            return f"""<|im_start|>system
You are a helpful assistant that answers questions using only the provided context. If the answer is not in the context, say you don't know. Provide the source filename and page number.
<|im_end|>
<|im_start|>user
Context:
{context}

Question: {user_query}<|im_end|>
<|im_start|>assistant
"""
        else:
            return f"""<|im_start|>system
You are a helpful assistant.
<|im_end|>
<|im_start|>user
{user_query}<|im_end|>
<|im_start|>assistant
"""



    def get_model_info(self) -> Dict:
        """Get information about the currently loaded model"""
        if self.model is None:
            return {"loaded": False}
        
        return {
            "loaded": True,
            "model_name": self.model_name,
            "model_path": str(self.model_path),
            "config": self.model_config
        }


class RAGWithLLM:
    """Complete RAG system with LLM integration"""
    
    def __init__(self):
        from rag_pipeline import RAGPipeline
        self.rag_pipeline = RAGPipeline()
        self.llm_loader = LLMLoader()
        self.model_loaded = False

    def initialize(self, model_name: str = "tinyllama") -> Dict:
        """Initialize the complete RAG + LLM system"""
        # Check GPU
        gpu_info = self.llm_loader.check_gpu_availability()
        
        # Load LLM model
        model_loaded = self.llm_loader.load_model(model_name)
        
        if model_loaded:
            self.model_loaded = True
            
        return {
            "gpu_info": gpu_info,
            "model_loaded": model_loaded,
            "rag_ready": True,
            "system_ready": model_loaded
        }

    def process_document(self, file_path: str) -> Dict:
        """Process and store a document"""
        return self.rag_pipeline.process_and_store_document(file_path)

    def answer_question(self, question: str) -> Dict:
        """Answer a question using RAG + LLM"""
        # Step 1: Retrieve relevant context
        rag_result = self.rag_pipeline.query(question)
        
        # Step 2: Generate answer using LLM
        if self.model_loaded:
            context = rag_result.get("retrieved_chunk")
            rag_prompt = self.llm_loader._format_rag_prompt(question, context)
            llm_response = self.llm_loader.generate_response(rag_prompt)
            
            if llm_response["error"]:
                answer = f"Retrieved relevant context but LLM generation failed: {llm_response['error']}"
            else:
                answer = llm_response["response"]
            
            return {
                "question": question,
                "answer": answer,
                "source": rag_result.get("source"),
                "similarity_score": rag_result.get("similarity_score"),
                "llm_used": True,
                "llm_metadata": llm_response.get("metadata"),
                "processing_time": rag_result.get("processing_time")
            }
        else:
            # Fallback to just returning the retrieved chunk
            return {
                "question": question,
                "answer": f"LLM not loaded. Retrieved context: {rag_result.get('retrieved_chunk')}",
                "source": rag_result.get("source"),
                "similarity_score": rag_result.get("similarity_score"),
                "llm_used": False
            }

    def get_system_status(self) -> Dict:
        """Get comprehensive system status"""
        return {
            "rag_stats": self.rag_pipeline.get_system_stats(),
            "llm_info": self.llm_loader.get_model_info(),
            "gpu_info": self.llm_loader.check_gpu_availability(),
            "system_ready": self.model_loaded
        }
