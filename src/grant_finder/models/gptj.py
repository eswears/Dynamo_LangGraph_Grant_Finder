from langchain.embeddings.base import Embeddings
import logging
from typing import List
from pathlib import Path
import numpy as np
from gpt4all import GPT4All

class GPTJ4AllEmbeddings(Embeddings):
    def __init__(self, model_path: str, logger: logging.Logger):
        self.logger = logger
        self.provider = "gptj"
        try:
            self.model = GPT4All(model_path)
            self.embedding_dim = 4096  # GPT-J's native embedding dimension
        except Exception as e:
            self.logger.error(f"Failed to initialize GPT-J model: {str(e)}")
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of documents"""
        embeddings = []
        for text in texts:
            try:
                embedding = self.model.generate_embedding(text)
                embeddings.append(embedding)
            except Exception as e:
                self.logger.error(f"Error generating embedding: {str(e)}")
                raise
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        try:
            return self.model.generate_embedding(text)
        except Exception as e:
            self.logger.error(f"Error generating query embedding: {str(e)}")
            raise

    def test_embeddings(model_path: str):
        """Test GPT-J embeddings functionality"""
        import logging
        logger = logging.getLogger('embedding_test')
        
        try:
            embeddings = GPTJ4AllEmbeddings(model_path, logger)
            
            # Test single embedding
            test_text = "This is a test sentence for embedding generation."
            single_embedding = embeddings.embed_query(test_text)
            print(f"Single embedding shape: {len(single_embedding)}")
            
            # Test batch embeddings
            test_texts = [
                "First test document",
                "Second test document",
                "Third test document"
            ]
            batch_embeddings = embeddings.embed_documents(test_texts)
            print(f"Batch embeddings shape: {len(batch_embeddings)}x{len(batch_embeddings[0])}")
            
            return True
            
        except Exception as e:
            logger.error(f"Embedding test failed: {str(e)}")
            return False

if __name__ == "__main__":
    #to test embeddings, run python -m grant_finder.models.gptj "C:/code/Dynamo/models/gptj/gptj-6b-ggml-q4.bin"
    import sys
    if len(sys.argv) != 2:
        print("Usage: python gptj.py <model_path>")
        sys.exit(1)
    
    success = GPTJ4AllEmbeddings.test_embeddings(sys.argv[1])
    sys.exit(0 if success else 1)