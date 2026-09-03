import logging
from pathlib import Path
from datasets import load_dataset

logger = logging.getLogger(__name__)

class UrduCorpusLoader:
    """Fetches and cleans citable Urdu text data (Wikimedia Wikipedia) for Continual Pre-Training."""
    
    def __init__(self, output_path: Path, max_samples: int = 10000):
        self.output_path = output_path
        self.max_samples = max_samples
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def fetch_and_clean(self):
        logger.info(f"Fetching Wikimedia Wikipedia Urdu dataset (limit={self.max_samples})...")
        
        # Wikimedia is officially distributed in pure Parquet format.
        # It requires no custom scripts, no tokens, and passes all security checks.
        dataset = load_dataset("wikimedia/wikipedia", "20231101.ur", split=f"train[:{self.max_samples}]")
        
        logger.info("Cleaning and formatting text...")
        cleaned_texts = []
        for item in dataset:
            text = item.get("text", "").replace("\n", " ").strip()
            
            # Keep only substantial paragraphs to ensure high-quality training data
            if len(text) > 100:
                cleaned_texts.append(text)
                
        with open(self.output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(cleaned_texts))
            
        logger.info(f"✅ Saved {len(cleaned_texts)} cleaned documents to {self.output_path}")
        logger.info("💡 Citation: 'Wikimedia Downloads' (Wikimedia Foundation, 2023).")