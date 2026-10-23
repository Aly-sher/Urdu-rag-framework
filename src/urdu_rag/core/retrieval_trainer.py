import logging
from pathlib import Path
from typing import List
import json
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

from urdu_rag.configs.schema import EmbeddingConfig

logger = logging.getLogger(__name__)

class UrduBiEncoderTrainer:
    """
    Trains a Bi-Encoder for Urdu retrieval using InfoNCE (MultipleNegativesRankingLoss).
    """
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initializing Bi-Encoder with {config.base_model_name}")
        self.model = SentenceTransformer(config.base_model_name)

    def load_triplets(self, filepath: Path) -> List[InputExample]:
        """Loads (query, positive, negative) triplets from a JSONL file."""
        examples = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                # InputExample expects a list of texts: [anchor, positive, negative]
                examples.append(InputExample(texts=[data['query'], data['positive'], data['negative']]))
        return examples

    def train(self, train_file: Path):
        """Executes the contrastive training loop."""
        logger.info(f"Loading training triplets from {train_file}...")
        train_examples = self.load_triplets(train_file)
        
        logger.info(f"Starting InfoNCE contrastive training on {len(train_examples)} examples...")
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=self.config.batch_size)
        
        # MultipleNegativesRankingLoss is the standard InfoNCE implementation for sentence transformers
        train_loss = losses.MultipleNegativesRankingLoss(self.model)
        
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=self.config.epochs,
            optimizer_params={"lr": self.config.learning_rate},
            show_progress_bar=True,
            output_path=str(self.config.output_dir)
        )
        logger.info(f"✅ Bi-Encoder trained and saved to {self.config.output_dir}")