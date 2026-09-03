import typer
import yaml
import logging
from pathlib import Path
from rich.logging import RichHandler

from urdu_rag.configs.schema import TypologyConfig, Phase2Config, TokenizerConfig, LoRAConfig
from urdu_rag.core.typology import TypologicalAligner
from urdu_rag.core.tokenization import NastaliqTokenizerTrainer
from urdu_rag.core.model_setup import LoRAModelInitializer
from urdu_rag.core.data_loader import UrduCorpusLoader
from transformers import PreTrainedTokenizerFast

# Setup professional logging with Rich for beautiful terminal output
logging.basicConfig(
    level="INFO", 
    format="%(message)s", 
    datefmt="[%X]", 
    handlers=[RichHandler(rich_tracebacks=True, markup=True)]
)

app = typer.Typer(help="Urdu RAG Framework: A typology-driven adaptation toolkit.")

def _load_yaml(config_path: Path) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

@app.command()
def fetch_data(
    output: Path = typer.Option("data/urdu_wikipedia_corpus.txt", help="Output file path"),
    samples: int = typer.Option(500, help="Number of articles to fetch (use large number for full paper)")
):
    """Fetch and clean the Urdu Wikipedia corpus for training."""
    loader = UrduCorpusLoader(output_path=output, max_samples=samples)
    loader.fetch_and_clean()

@app.command()
def align_typology(
    config: Path = typer.Option("configs/phase1_typology.yaml", help="Path to typology config")
):
    """Phase 1: Compute URIEL+ typological distances and Procrustes alignment."""
    raw = _load_yaml(config)
    cfg = TypologyConfig(**raw)
    aligner = TypologicalAligner(cfg)
    best_anchor = aligner.run_pipeline()
    typer.echo(f"Pipeline complete. Best anchor: {best_anchor}")

@app.command()
def train_tokenizer(
    config: Path = typer.Option("configs/phase2_tokenizer_lora.yaml", help="Path to phase 2 config")
):
    """Phase 2A: Train the custom Nastaliq BPE Tokenizer."""
    raw = _load_yaml(config)
    cfg = TokenizerConfig(**raw['tokenizer'])
    trainer = NastaliqTokenizerTrainer(cfg)
    trainer.train()

@app.command()
def setup_lora(
    config: Path = typer.Option("configs/phase2_tokenizer_lora.yaml", help="Path to phase 2 config")
):
    """Phase 2B: Initialize Base Model, resize embeddings, and inject LoRA."""
    raw = _load_yaml(config)
    tok_cfg = TokenizerConfig(**raw['tokenizer'])
    lora_cfg = LoRAConfig(**raw['lora'])
    
    # Load the tokenizer we just trained
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tok_cfg.output_dir))
    
    initializer = LoRAModelInitializer(lora_cfg)
    initializer.initialize(tokenizer)

if __name__ == "__main__":
    app()