import logging
import numpy as np
import torch
from pathlib import Path
from scipy.linalg import orthogonal_procrustes
import lang2vec.lang2vec as l2v

from urdu_rag.configs.schema import TypologyConfig

logger = logging.getLogger(__name__)

class TypologicalAligner:
    """
    Computes typological distances and Procrustes alignment matrices 
    for low-resource language adaptation using the URIEL+ database.
    """
    
    def __init__(self, config: TypologyConfig):
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized TypologicalAligner for target: {config.target_lang}")

    def _fetch_and_clean_vector(self, lang_code: str, feature: str) -> np.ndarray:
        """Fetches a URIEL+ feature vector and handles missing values ('--')."""
        res = l2v.get_features([lang_code], [feature])
        raw_vec = res[lang_code]
        # Replace missing values with 0.0 and cast to float32
        clean_vec = [0.0 if str(v) == '--' else float(v) for v in raw_vec]
        return np.array(clean_vec, dtype=np.float32)

    def compute_distances(self) -> dict:
        """Calculates Euclidean distances between target and anchor languages."""
        logger.info("Fetching URIEL+ typological vectors...")
        vectors = {}
        
        all_langs = [self.config.target_lang] + self.config.anchor_langs
        for lang in all_langs:
            vectors[lang] = {}
            for feature in self.config.feature_sets:
                vectors[lang][feature] = self._fetch_and_clean_vector(lang, feature)

        distances = {}
        target_vec = np.concatenate([vectors[self.config.target_lang][f] for f in self.config.feature_sets])
        
        for anchor in self.config.anchor_langs:
            anchor_vec = np.concatenate([vectors[anchor][f] for f in self.config.feature_sets])
            dist = float(np.linalg.norm(target_vec - anchor_vec))
            distances[anchor] = dist
            logger.info(f"Distance to {anchor}: {dist:.4f}")
            
        return distances

    def compute_procrustes_matrix(self, anchor_lang: str) -> torch.Tensor:
        """
        Computes the orthogonal transformation matrix W to align the target 
        language's typological space to the specified anchor language.
        """
        logger.info(f"Computing Procrustes alignment for {self.config.target_lang} -> {anchor_lang}")
        
        # Fetch combined vectors
        X_raw = np.concatenate([self._fetch_and_clean_vector(self.config.target_lang, f) for f in self.config.feature_sets])
        Y_raw = np.concatenate([self._fetch_and_clean_vector(anchor_lang, f) for f in self.config.feature_sets])
        
        # Reshape for Scipy's orthogonal_procrustes (requires 2D arrays)
        X = X_raw.reshape(1, -1)
        Y = Y_raw.reshape(1, -1)
        
        W, disparity = orthogonal_procrustes(X, Y)
        logger.info(f"Procrustes matrix computed. Disparity (error): {disparity:.4f}")
        
        # Save the matrix
        W_tensor = torch.tensor(W, dtype=torch.float32)
        save_path = self.config.output_dir / f"procrustes_W_{self.config.target_lang}_{anchor_lang}.pt"
        torch.save(W_tensor, save_path)
        logger.info(f"Saved Procrustes matrix to {save_path}")
        
        return W_tensor

    def run_pipeline(self) -> str:
        """Executes the full typological profiling and alignment pipeline."""
        distances = self.compute_distances()
        best_anchor = min(distances, key=distances.get)
        logger.info(f"Closest typological anchor identified: {best_anchor}")
        
        self.compute_procrustes_matrix(best_anchor)
        return best_anchor