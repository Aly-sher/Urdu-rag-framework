import logging
from pathlib import Path
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

from urdu_rag.configs.schema import TokenizerConfig

logger = logging.getLogger(__name__)

class NastaliqTokenizerTrainer:
    """Trains a Byte-Level BPE tokenizer optimized for the Nastaliq script."""
    
    def __init__(self, config: TokenizerConfig):
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def train(self) -> PreTrainedTokenizerFast:
        if not self.config.corpus_path.exists():
            raise FileNotFoundError(f"Corpus not found at {self.config.corpus_path}. Run data_loader first.")

        logger.info(f"Training BPE tokenizer (vocab_size={self.config.vocab_size}) on {self.config.corpus_path}...")
        
        tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        
        trainer = trainers.BpeTrainer(
            vocab_size=self.config.vocab_size,
            special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
        )
        
        tokenizer.train([str(self.config.corpus_path)], trainer)
        
        # Save raw tokenizer
        raw_path = self.config.output_dir / "tokenizer.json"
        tokenizer.save(str(raw_path))
        
        # Wrap in HuggingFace Fast Tokenizer
        hf_tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(raw_path),
            bos_token="[BOS]", eos_token="[EOS]",
            unk_token="[UNK]", pad_token="[PAD]"
        )
        # Save the HuggingFace format for easy loading later
        hf_tokenizer.save_pretrained(str(self.config.output_dir))
        
        logger.info(f"✅ Tokenizer trained and saved to {self.config.output_dir}")
        return hf_tokenizer