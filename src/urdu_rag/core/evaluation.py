import logging
import numpy as np
from typing import List, Dict

logger = logging.getLogger(__name__)

class RetrievalEvaluator:
    """
    Computes standard Information Retrieval metrics: Recall@k, MRR, and nDCG@k.
    """
    def __init__(self, k_values: List[int] = [1, 3, 5, 10]):
        self.k_values = k_values

    def compute_recall_at_k(self, retrieved_ids: List[int], relevant_ids: List[int], k: int) -> float:
        top_k = retrieved_ids[:k]
        relevant_set = set(relevant_ids)
        hits = len(set(top_k).intersection(relevant_set))
        return hits / len(relevant_set) if relevant_set else 0.0

    def compute_mrr(self, retrieved_ids: List[int], relevant_ids: List[int]) -> float:
        relevant_set = set(relevant_ids)
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_set:
                return 1.0 / (i + 1)
        return 0.0

    def compute_dcg_at_k(self, relevances: List[float], k: int) -> float:
        relevances = relevances[:k]
        dcg = 0.0
        for i, rel in enumerate(relevances):
            # Standard DCG formula: (2^rel - 1) / log2(i + 2)
            dcg += (2**rel - 1) / np.log2(i + 2)
        return dcg

    def compute_ndcg_at_k(self, retrieved_relevances: List[float], ideal_relevances: List[float], k: int) -> float:
        dcg = self.compute_dcg_at_k(retrieved_relevances, k)
        idcg = self.compute_dcg_at_k(sorted(ideal_relevances, reverse=True), k)
        return dcg / idcg if idcg > 0 else 0.0

    def evaluate(self, predictions: Dict[str, List[int]], ground_truth: Dict[str, List[int]], relevances_map: Dict[str, Dict[int, float]] = None) -> Dict[str, float]:
        """
        Evaluates the retrieval pipeline.
        predictions: {query_id: [retrieved_doc_ids]}
        ground_truth: {query_id: [relevant_doc_ids]}
        relevances_map: {query_id: {doc_id: relevance_score}} (for nDCG)
        """
        results = {f"Recall@{k}": 0.0 for k in self.k_values}
        results["MRR"] = 0.0
        if relevances_map:
            for k in self.k_values:
                results[f"nDCG@{k}"] = 0.0

        num_queries = len(ground_truth)
        if num_queries == 0:
            return results
            
        for q_id, relevant_ids in ground_truth.items():
            retrieved_ids = predictions.get(q_id, [])
            
            # 1. Recall@k
            for k in self.k_values:
                results[f"Recall@{k}"] += self.compute_recall_at_k(retrieved_ids, relevant_ids, k)
                
            # 2. MRR
            results["MRR"] += self.compute_mrr(retrieved_ids, relevant_ids)
            
            # 3. nDCG@k
            if relevances_map and q_id in relevances_map:
                rel_map = relevances_map[q_id]
                retrieved_rels = [rel_map.get(doc_id, 0.0) for doc_id in retrieved_ids]
                ideal_rels = [rel_map.get(doc_id, 0.0) for doc_id in relevant_ids]
                
                for k in self.k_values:
                    results[f"nDCG@{k}"] += self.compute_ndcg_at_k(retrieved_rels, ideal_rels, k)

        # Average over all queries
        for key in results:
            results[key] /= num_queries
            
        return results