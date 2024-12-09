# verification_script.py
import logging
from pathlib import Path
import numpy as np
from typing import List
import time

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0
    return np.dot(a, b) / (norm_a * norm_b)

def verify_gptj_embeddings(model_path: str):
    """Verify GPT-J embeddings are working correctly"""
    from grant_finder.models.gptj import GPTJ4AllEmbeddings
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('embedding_verification')
    
    try:
        logger.info("Initializing GPT-J embeddings...")
        embeddings = GPTJ4AllEmbeddings(model_path, logger)
        
        # Test 1: Basic Functionality
        logger.info("\nTest 1: Basic Functionality")
        test_text = "This is a test sentence."
        embedding = embeddings.embed_query(test_text)
        logger.info(f"✓ Generated embedding of dimension: {len(embedding)}")
        logger.info(f"✓ Embedding values range: [{min(embedding):.3f}, {max(embedding):.3f}]")
        
        # Test 2: Semantic Similarity
        logger.info("\nTest 2: Semantic Similarity")
        similar_pairs = [
            ("The company develops AI solutions", "They create artificial intelligence software"),
            ("The project needs funding", "This initiative requires financial support"),
            ("Military defense applications", "Defense department use cases")
        ]
        
        different_pairs = [
            ("The company develops AI solutions", "The weather is nice today"),
            ("The project needs funding", "Cats are cute animals"),
            ("Military defense applications", "Recipe for chocolate cake")
        ]
        
        logger.info("Testing similar sentence pairs...")
        for text1, text2 in similar_pairs:
            emb1 = embeddings.embed_query(text1)
            emb2 = embeddings.embed_query(text2)
            similarity = cosine_similarity(emb1, emb2)
            logger.info(f"Similarity ({text1} | {text2}): {similarity:.3f}")
            if similarity < 0.5:
                logger.warning(f"⚠️ Low similarity for supposedly similar texts: {similarity:.3f}")
            else:
                logger.info("✓ Similar texts have high similarity score")
        
        logger.info("\nTesting different sentence pairs...")
        for text1, text2 in different_pairs:
            emb1 = embeddings.embed_query(text1)
            emb2 = embeddings.embed_query(text2)
            similarity = cosine_similarity(emb1, emb2)
            logger.info(f"Similarity ({text1} | {text2}): {similarity:.3f}")
            if similarity > 0.7:
                logger.warning(f"⚠️ High similarity for supposedly different texts: {similarity:.3f}")
            else:
                logger.info("✓ Different texts have low similarity score")
        
        # Test 3: Batch Processing
        logger.info("\nTest 3: Batch Processing")
        batch_texts = [
            "First test document",
            "Second test document",
            "Third test document"
        ]
        start_time = time.time()
        batch_embeddings = embeddings.embed_documents(batch_texts)
        end_time = time.time()
        logger.info(f"✓ Successfully processed batch of {len(batch_texts)} documents")
        logger.info(f"✓ Time per document: {(end_time - start_time) / len(batch_texts):.2f} seconds")
        
        # Test 4: Consistency
        logger.info("\nTest 4: Embedding Consistency")
        test_text = "This is a consistency test."
        emb1 = embeddings.embed_query(test_text)
        emb2 = embeddings.embed_query(test_text)
        consistency = cosine_similarity(emb1, emb2)
        logger.info(f"Same text similarity (should be near 1.0): {consistency:.3f}")
        if consistency < 0.99:
            logger.warning(f"⚠️ Embeddings not consistent for same text: {consistency:.3f}")
        else:
            logger.info("✓ Embeddings are consistent for same text")
        
        return True
        
    except Exception as e:
        logger.error(f"Verification failed: {str(e)}")
        return False

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python verification_script.py <model_path>")
        sys.exit(1)
    
    success = verify_gptj_embeddings(sys.argv[1])
    sys.exit(0 if success else 1)