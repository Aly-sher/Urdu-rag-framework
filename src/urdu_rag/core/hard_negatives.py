import logging
import numpy as np
from typing import List, Tuple
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

class BM25HardNegativeMiner:
    """
    Mines hard negatives for contrastive retrieval training.
    Hard negatives are documents that share high lexical overlap (BM25) 
    with the query but are semantically incorrect, forcing the Bi-Encoder 
    to learn deep semantic boundaries rather than keyword matching.
    """
    
    def __init__(self, corpus: List[str]):
        logger.info(f"Initializing BM25 index over {len(corpus)} documents...")
        # Tokenize corpus for BM25 (simple whitespace split for Urdu)
        self.tokenized_corpus = [doc.split(" ") for doc in corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self.corpus = corpus

    def mine_triplets(self, queries: List[str], positives: List[str], k_negatives: int = 3) -> List[Tuple[str, str, str]]:
        """
        Generates (Query, Positive, Hard Negative) triplets for InfoNCE training.
        """
        logger.info(f"Mining hard negatives for {len(queries)} queries...")
        triplets = []
        
        tokenized_queries = [q.split(" ") for q in queries]
        
        for i, query in enumerate(queries):
            # Get BM25 scores for this query against the whole corpus
            scores = self.bm25.get_scores(tokenized_queries[i])
            
            # Get indices of top-k lexically similar documents
            # We ask for k_negatives + 1 just in case the positive document is in the top-k
            top_k_indices = np.argsort(scores)[::-1][:k_negatives + 1]
            
            positive_text = positives[i]
            negatives_added = 0
            
            for idx in top_k_indices:
                candidate_neg = self.corpus[idx]
                # Ensure the hard negative is not actually the positive document
                if candidate_neg != positive_text and negatives_added < k_negatives:
                    triplets.append((query, positive_text, candidate_neg))
                    negatives_added += 1
                    
        logger.info(f"✅ Mined {len(triplets)} contrastive triplets.")
        return triplets