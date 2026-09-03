"""
Strict configuration schemas for the Urdu RAG Framework.
Uses Pydantic to ensure type safety and validate all parameters before execution.
"""
from pydantic import BaseModel, Field, FilePath, DirectoryPath
from typing import List, Optional
from pathlib import Path


class TypologyConfig(BaseModel):
    """Configuration for Phase 1: Typological Alignment and Procrustes Matrix."""
    target_lang: str = Field(default="urd", description="ISO 639-3 code for the target language")
    anchor_langs: List[str] = Field(
        default=["hin", "fas", "eng"], 
        description="List of ISO 639-3 codes for candidate anchor languages"
    )
    feature_sets: List[str] = Field(
        default=["syntax_average", "phonology_average"], 
        description="URIEL+ feature sets to extract for distance calculation"
    )
    output_dir: Path = Field(default=Path("data/typology"), description="Directory to save Procrustes matrices")


class TokenizerConfig(BaseModel):
    """Configuration for Phase 2A: Custom BPE Tokenizer Training."""
    corpus_path: FilePath = Field(description="Absolute or relative path to the raw Urdu text corpus")
    vocab_size: int = Field(default=32000, description="Target vocabulary size for the BPE tokenizer", ge=1000)
    output_dir: Path = Field(default=Path("data/tokenizer"), description="Directory to save the trained tokenizer")


class LoRAConfig(BaseModel):
    """Configuration for Phase 2B: LoRA Adapter Initialization."""
    base_model_name: str = Field(default="Qwen/Qwen2.5-0.5B", description="Hugging Face model ID for the base model")
    lora_r: int = Field(default=16, description="Rank of the LoRA update matrices", ge=1)
    lora_alpha: int = Field(default=32, description="Scaling factor for LoRA weights", ge=1)
    lora_dropout: float = Field(default=0.05, description="Dropout probability for LoRA layers", ge=0.0, le=1.0)
    target_modules: List[str] = Field(
        default=["q_proj", "v_proj", "k_proj", "o_proj"], 
        description="Names of the transformer modules to apply LoRA to"
    )
    output_dir: Path = Field(default=Path("data/lora_model"), description="Directory to save the LoRA-adapted model")


class Phase2Config(BaseModel):
    """Wrapper config for Phase 2 to load both Tokenizer and LoRA configs from one YAML."""
    tokenizer: TokenizerConfig
    lora: LoRAConfig