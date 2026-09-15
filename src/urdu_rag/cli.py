import typer
import yaml
import json
import logging
from pathlib import Path
from rich.logging import RichHandler
from transformers import PreTrainedTokenizerFast

from urdu_rag.configs.schema import TypologyConfig, TokenizerConfig, LoRAConfig
from urdu_rag.core.typology import TypologicalAligner
from urdu_rag.core.tokenization import NastaliqTokenizerTrainer
from urdu_rag.core.model_setup import LoRAModelInitializer
from urdu_rag.core.data_loader import UrduCorpusLoader
from urdu_rag.core.hard_negatives import BM25HardNegativeMiner
from urdu_rag.core.evaluation import RetrievalEvaluator

# Setup professional logging with Rich for beautiful terminal output
logging.basicConfig(
    level="INFO", 
    format="%(message)s", 
    handlers=[RichHandler(rich_tracebacks=True, markup=True)]
)
logger = logging.getLogger(__name__) # <--- THIS IS THE MISSING LINE

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

from urdu_rag.core.hard_negatives import BM25HardNegativeMiner
from urdu_rag.core.evaluation import RetrievalEvaluator

@app.command()
def mine_hard_negatives(
    corpus_file: Path = typer.Option("data/urdu_wikipedia_corpus.txt", help="Path to the cleaned corpus"),
    queries_file: Path = typer.Option("data/queries.jsonl", help="Path to query/positive pairs"),
    output_file: Path = typer.Option("data/training_triplets.jsonl", help="Where to save the mined triplets"),
    k_negatives: int = typer.Option(3, help="Number of hard negatives per query")
):
    """Phase 5A: Mine BM25 hard negatives for contrastive training."""
    logger.info("Loading corpus for BM25 indexing...")
    with open(corpus_file, 'r', encoding='utf-8') as f:
        corpus = [line.strip() for line in f if line.strip()]
        
    # NOTE: In a real run, you would load your actual Urdu queries/positives from a JSONL file.
    # For this architectural proof, we use dummy data to prove the pipeline works.
    queries = ["پاکستان کا دارالحکومت کون سا شہر ہے؟", "لاہور کس صوبے کا دارالحکومت ہے؟"]
    positives = ["اسلام آباد پاکستان کا دارالحکومت ہے۔", "لاہور پنجاب کا دارالحکومت اور ثقافتی مرکز ہے۔"]
    
    miner = BM25HardNegativeMiner(corpus)
    triplets = miner.mine_triplets(queries, positives, k_negatives=k_negatives)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for q, p, n in triplets:
            f.write(json.dumps({"query": q, "positive": p, "negative": n}, ensure_ascii=False) + "\n")
            
    logger.info(f"✅ Saved {len(triplets)} triplets to {output_file}")


@app.command()
def evaluate_retrieval(
    predictions_file: Path = typer.Option("data/predictions.json", help="Retriever outputs"),
    ground_truth_file: Path = typer.Option("data/ground_truth.json", help="Relevant document IDs")
):
    """Phase 5B: Evaluate retrieval performance using Recall@k, MRR, and nDCG@k."""
    logger.info("Initializing Retrieval Evaluator...")
    evaluator = RetrievalEvaluator(k_values=[1, 3, 5, 10])
    
    # NOTE: In a real run, load actual JSON files. Using dummy data for proof of concept.
    predictions = {"q1": [101, 102, 105], "q2": [201, 202]}
    ground_truth = {"q1": [102], "q2": [201]}
    relevances = {"q1": {102: 2.0, 101: 0.0}, "q2": {201: 2.0, 202: 0.0}}
    
    metrics = evaluator.evaluate(predictions, ground_truth, relevances)
    
    typer.echo("\n🏆 Retrieval Evaluation Metrics:")
    for metric, value in metrics.items():
        typer.echo(f"   {metric}: {value:.4f}")    

if __name__ == "__main__":
    app()