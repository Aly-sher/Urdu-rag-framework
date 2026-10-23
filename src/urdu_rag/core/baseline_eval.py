import logging
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class BaselineRetrievalEval:
    """
    Evaluates an off-the-shelf multilingual model to establish a baseline.
    Proves that without our InfoNCE hard-negative training, the model struggles with Urdu semantics.
    """
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        logger.info(f"Loading off-the-shelf baseline model: {model_name}")
        self.model = SentenceTransformer(model_name)

    def evaluate_mrr(self, data_file: Path) -> float:
        """Computes Mean Reciprocal Rank (MRR) for the baseline model."""
        queries = []
        positives = []
        negatives = []
        
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                queries.append(data['query'])
                positives.append(data['positive'])
                negatives.append(data['negative'])
                
        logger.info(f"Evaluating baseline on {len(queries)} queries...")
        
        # Encode all texts
        q_emb = self.model.encode(queries, convert_to_tensor=True)
        p_emb = self.model.encode(positives, convert_to_tensor=True)
        n_emb = self.model.encode(negatives, convert_to_tensor=True)
        
        # Normalize embeddings for cosine similarity
        q_emb = F.normalize(q_emb, p=2, dim=1)
        p_emb = F.normalize(p_emb, p=2, dim=1)
        n_emb = F.normalize(n_emb, p=2, dim=1)
        
        # Calculate similarities
        pos_scores = torch.sum(q_emb * p_emb, dim=1)
        neg_scores = torch.sum(q_emb * n_emb, dim=1)
        
        reciprocal_ranks = []
        for i in range(len(queries)):
            # If the positive doc scores higher than the negative doc, it's ranked #1 (RR = 1.0)
            # If the negative doc scores higher, the positive is ranked #2 (RR = 0.5)
            if pos_scores[i] > neg_scores[i]:
                reciprocal_ranks.append(1.0)
            else:
                reciprocal_ranks.append(0.5)
                
        mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
        return mrr