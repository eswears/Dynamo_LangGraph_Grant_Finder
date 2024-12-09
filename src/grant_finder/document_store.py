from pathlib import Path
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
import json
import os
from dotenv import load_dotenv
from grant_finder.models.bitnet import BitNetEmbeddings
from grant_finder.models.bitnet import BitNetLLM
from datetime import datetime

load_dotenv()

@dataclass
class DocumentLayer:
    """Represents a layer in the hierarchical document store"""
    name: str
    chunk_size: int
    chunk_overlap: int
    vector_store: Optional[FAISS] = None
    summaries: Dict[str, str] = None
    metadata: Dict[str, Any] = None

class HierarchicalDocumentStore:
    """Manages document storage and retrieval across multiple abstraction layers"""
    
    def __init__(self, base_path: Path, logger: logging.Logger, config: Dict, llm: Optional[BitNetLLM] = None, previous_context: Optional[Dict[str, Any]] = None):
        self.base_path = base_path
        self.logger = logger
        self.config = config
        
        # Check previous embeddings provider if context exists
        previous_provider = None
        if previous_context and "embedding_provider" in previous_context:
            previous_provider = previous_context["embedding_provider"]
        
        # Get current config
        current_provider = config["embeddings"]["provider"]
        self.embeddings_base_path = Path(config["embeddings"]["save_path"])
        
        # If providers don't match, prompt user
        if previous_provider and previous_provider != current_provider:
            response = input(f"Warning: Previous embeddings used {previous_provider} but current config specifies {current_provider}.\n"
                            f"Enter 'reprocess' to create new embeddings or 'switch' to use {previous_provider}: ")
            if response.lower() == "reprocess":
                # Clear previous context to force reprocessing
                previous_context = None
            elif response.lower() == "switch":
                current_provider = previous_provider
            else:
                raise ValueError("Invalid response. Must be 'reprocess' or 'switch'")
        
        # Initialize embeddings based on provider
        if current_provider == "openai":
            self.embeddings = OpenAIEmbeddings(
                model=config["embeddings"]["openai"]["model"],
                # Remove cache_size as it's not supported in current version
                model_kwargs={}
            )
        elif current_provider == "gptj":
            model_path = config["embeddings"]["gptj"]["model_path"]
            self.embeddings = GPTJ4AllEmbeddings(
                model_path=model_path,
                logger=logger
            )
        else:
            raise ValueError(f"Unknown embeddings provider: {current_provider}")
        
        # Store provider for future reference
        self.embedding_provider = current_provider
        
        # Define the hierarchical layers using env vars
        self.layers = {
            "high": DocumentLayer(
                "high", 
                int(os.getenv('HIGH_LEVEL_CHUNK_SIZE', 3000)),
                int(float(os.getenv('CHUNK_OVERLAP_RATIO', 0.1)) * int(os.getenv('HIGH_LEVEL_CHUNK_SIZE', 3000)))
            ),
            "mid": DocumentLayer(
                "mid", 
                int(os.getenv('MID_LEVEL_CHUNK_SIZE', 1500)),
                int(float(os.getenv('CHUNK_OVERLAP_RATIO', 0.1)) * int(os.getenv('MID_LEVEL_CHUNK_SIZE', 1500)))
            ),
            "low": DocumentLayer(
                "low", 
                int(os.getenv('LOW_LEVEL_CHUNK_SIZE', 500)),
                int(float(os.getenv('CHUNK_OVERLAP_RATIO', 0.1)) * int(os.getenv('LOW_LEVEL_CHUNK_SIZE', 500)))
            )
        }
        
        # Restore previous context if available
        if previous_context:
            self._restore_context(previous_context)
        
        # Initialize document storage
        self._initialize_stores()
    
    def _initialize_stores(self):
        """Initialize vector stores for each layer"""
        store_path = self.embeddings_base_path / self.embedding_provider
        store_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize empty stores for each layer if they don't exist
        for layer_name, layer in self.layers.items():
            layer_path = store_path / layer_name
            # Try to load existing store first
            if layer_path.exists():
                try:
                    layer.vector_store = FAISS.load_local(
                        str(layer_path), 
                        self.embeddings,
                        allow_dangerous_deserialization=True
                    )
                    self.logger.info(f"Loaded existing {layer_name} layer store from {layer_path}")
                except Exception as e:
                    self.logger.warning(f"Could not load existing store for {layer_name}: {str(e)}")
                    layer.vector_store = None
            
            # Create new store if needed
            if layer.vector_store is None:
                layer.vector_store = FAISS.from_texts(
                    ["initialization"], 
                    self.embeddings,
                    metadatas=[{"source": "initialization"}]
                )
                # Save immediately after creation
                try:
                    layer_path.parent.mkdir(parents=True, exist_ok=True)
                    layer.vector_store.save_local(str(layer_path))
                    self.logger.info(f"Created new {layer_name} layer store at {layer_path}")
                except Exception as e:
                    self.logger.error(f"Error saving {layer_name} layer store: {str(e)}")

            if not layer_path.exists():
                layer.vector_store.save_local(str(layer_path))
                self.logger.info(f"Created new {layer_name} layer store at {layer_path}")
        
        # Check for existing embeddings metadata
        metadata_path = store_path / "embeddings_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                stored_metadata = json.load(f)
                stored_provider = stored_metadata.get("provider")
                
                if stored_provider != self.embedding_provider:
                    response = input(
                        f"\nWarning: Existing embeddings were created using {stored_provider} but current config specifies {self.embedding_provider}.\n"
                        f"Options:\n"
                        f"1. 'reprocess' - Create new embeddings with {self.embedding_provider}\n"
                        f"2. 'switch' - Switch to using {stored_provider}\n"
                        f"Enter choice (1/2): "
                    )
                    
                    if response == "1":
                        self.logger.info(f"Reprocessing with {self.embedding_provider}")
                        # Delete existing stores to force reprocessing
                        for layer_name in self.layers:
                            layer_path = store_path / layer_name
                            if layer_path.exists():
                                import shutil
                                shutil.rmtree(layer_path)
                    elif response == "2":
                        self.logger.info(f"Switching to {stored_provider}")
                        self.embedding_provider = stored_provider
                        # Reinitialize embeddings with stored provider
                        if stored_provider == "openai":
                            self.embeddings = OpenAIEmbeddings(
                                cache_size=os.getenv('EMBEDDINGS_CACHE_SIZE', '1G')
                            )
                        elif stored_provider == "bitnet":
                            if not hasattr(self, 'llm') or not self.llm:
                                raise ValueError("BitNet LLM required for BitNet embeddings")
                            self.embeddings = BitNetEmbeddings(self.llm.model_config_, self.logger)
                        # Update store path for switched provider
                        store_path = self.embeddings_base_path / self.embedding_provider
                        store_path.mkdir(parents=True, exist_ok=True)
                    else:
                        raise ValueError("Invalid choice. Must be '1' or '2'")

        for layer_name, layer in self.layers.items():
            layer_path = store_path / layer_name
            if layer_path.exists():
                layer.vector_store = FAISS.load_local(
                    str(layer_path), 
                    self.embeddings,
                    allow_dangerous_deserialization=True  # Only enable this for trusted, local files
                )
                self.logger.info(f"Loaded existing {layer_name} layer store from {layer_path}")
            else:
                self.logger.info(f"Will create new {layer_name} layer store at {layer_path}")
        
        # Save current embeddings metadata
        with open(metadata_path, 'w') as f:
            json.dump({
                "provider": self.embedding_provider,
                "timestamp": datetime.now().isoformat()
            }, f, indent=2)
    
    def _create_chunks(self, documents: List[Document], layer: DocumentLayer) -> List[Document]:
        """Create chunks for a specific layer"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=layer.chunk_size,
            chunk_overlap=layer.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        return splitter.split_documents(documents)
    
    def _create_summary(self, chunks: List[Document], level: str) -> str:
        """Create a summary for a group of chunks"""
        # Combine chunk content
        combined_text = "\n\n".join(chunk.page_content for chunk in chunks)
        
        # Create prompt based on level
        if level == "high":
            prompt = """Create a high-level executive summary of these documents, focusing on:
            - Key company capabilities
            - Main technical focus areas
            - Overall strategic direction
            
            Keep the summary concise and focused on the most important points."""
            
        elif level == "mid":
            prompt = """Create a detailed technical summary of these documents, focusing on:
            - Specific technical capabilities
            - Implementation details
            - Project examples and use cases
            
            Include important technical details while maintaining clarity."""
            
        else:  # low level
            return combined_text  # Return raw content for lowest level
        
        # Get embeddings for the summary
        summary_embedding = self.embeddings.embed_query(combined_text)
        
        # Store the summary with its embedding
        if level != "low":
            layer = self.layers[level]
            if layer.summaries is None:
                layer.summaries = {}
            
            # Generate unique ID for this summary
            summary_id = f"summary_{len(layer.summaries)}"
            layer.summaries[summary_id] = {
                "text": combined_text,
                "embedding": summary_embedding
            }
        
        return combined_text
    
    def process_documents(self, documents: List[Document]):
        """Process documents into hierarchical layers"""
        total_layers = len(self.layers)
        processed_layers = 0
        
        try:
            for layer_name, layer in self.layers.items():
                self.logger.info(f"Processing {layer_name} layer ({processed_layers + 1}/{total_layers})...")
                
                total_chunks = 0
                processed_chunks = 0
                
                try:
                    chunks = self._create_chunks(documents, layer)
                    total_chunks = len(chunks)
                    
                    # Create vector store if needed
                    if layer.vector_store is None:
                        layer.vector_store = FAISS.from_documents(chunks, self.embeddings)
                        layer.vector_store.save_local(str(self.base_path / ".document_store" / layer_name))
                    else:
                        layer.vector_store.add_documents(chunks)
                    
                    # Create summaries for chunks
                    if layer_name != "low":  # Don't summarize lowest layer
                        layer.summaries = {}
                        for i, chunk_group in enumerate(chunks):
                            try:
                                summary = self._create_summary([chunk_group], layer_name)
                                layer.summaries[f"group_{i}"] = summary
                                processed_chunks += 1
                                self.logger.debug(f"Processed chunk {processed_chunks}/{total_chunks} in {layer_name} layer")
                            except Exception as e:
                                self.logger.error(f"Error creating summary for chunk {i} in {layer_name} layer: {str(e)}")
                                continue
                    
                    processed_layers += 1
                    self.logger.info(f"Successfully processed {layer_name} layer ({processed_layers}/{total_layers})")
                    
                except Exception as e:
                    self.logger.error(f"Error processing {layer_name} layer: {str(e)}")
                    continue
                
        except Exception as e:
            self.logger.error(f"Error in document processing: {str(e)}")
            raise
    
    def query(self, query: str, layer: str = "high", k: int = 4) -> List[Document]:
        """Query a specific layer of the document store"""
        if layer not in self.layers:
            raise ValueError(f"Invalid layer: {layer}")
            
        layer_store = self.layers[layer].vector_store
        if layer_store is None:
            raise RuntimeError(f"Layer {layer} not initialized")
            
        return layer_store.similarity_search(query, k=k)
    
    def multi_layer_query(self, query: str, k: int = 4) -> Dict[str, List[Document]]:
        """Query all layers and return combined results"""
        results = {}
        for layer_name in self.layers.keys():
            results[layer_name] = self.query(query, layer=layer_name, k=k)
        return results
    
    def _restore_context(self, context: Optional[Dict[str, Any]]) -> None:
        """Restore previous context to layers"""
        if context is None or "layer_states" not in context:
            return
            
        for layer_name, layer_state in context["layer_states"].items():
            if layer_name in self.layers:
                self.layers[layer_name].summaries = layer_state.get("summaries", {})
                self.layers[layer_name].metadata = layer_state.get("metadata", {})
    
    def cleanup(self):
        """Clean up resources"""
        try:
            for layer in self.layers.values():
                if layer.vector_store is not None:
                    # Save store before clearing
                    layer_path = self.embeddings_base_path / self.embedding_provider / layer.name
                    layer.vector_store.save_local(str(layer_path))
                    # Clear references
                    layer.vector_store = None
                    layer.summaries = {}
            
            # Clear embeddings model
            if hasattr(self.embeddings, 'model'):
                if hasattr(self.embeddings.model, 'cleanup'):
                    self.embeddings.model.cleanup()
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Save context to temporary file using provider-specific path
            store_path = self.embeddings_base_path / self.embedding_provider
            temp_context_path = store_path / "temp_context.json"
            with open(temp_context_path, 'w') as f:
                json.dump(context, f, indent=2)
            
            # Cleanup resources
            for layer in self.layers.values():
                if layer.vector_store is not None:
                    layer_path = store_path / layer.name
                    layer.vector_store.save_local(str(layer_path))
                    layer.vector_store = None
            
            # Clear embeddings and model resources
            if hasattr(self.embeddings, 'model'):
                if hasattr(self.embeddings.model, 'cleanup'):
                    self.embeddings.model.cleanup()
                if hasattr(self.embeddings.model, 'reset'):
                    self.embeddings.model.reset()
                    
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")