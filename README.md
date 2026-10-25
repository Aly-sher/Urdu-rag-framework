# Urdu-RAG Framework
### A Typology-Driven Retrieval-Augmented Generation Pipeline for Low-Resource Languages

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> **Project Status:** Active Research Prototype.  
> This repository implements the architectural scaffolding for a typology-informed RAG system for Urdu. While the core modules (Typology, Tokenization, Retrieval, Evaluation) are implemented, several components (such as full WECHSEL initialization and large-scale CPT) are currently in the "Smoke Test" or "Planned" phases. Please see the [Implementation Status](#-implementation-status) table below for details.

---

## 🏛️ Implementation Status

To ensure transparency and reproducibility, we explicitly categorize the current state of each pipeline component.

| Component | Status | Current Implementation & Limitations |
| :--- | :--- | :--- |
| **Typological Alignment** | 🟡 Partial | URIEL+ distance computation is implemented. **Planned:** Proper WECHSEL-style token-level embedding initialization. |
| **Data Decontamination** | 🟡 Partial | `UrduDataProcessor` (n-gram filtering) logic is implemented. **Planned:** Full CLI integration and normalization pipeline. |
| **LoRA CPT** | 🔴 Smoke Test | Currently runs a 10-step smoke test to verify the training loop. **Planned:** Full epoch training and model checkpointing. |
| **BM25 Hard-Negative Mining** | 🟢 Implemented | Functional, but requires noise reduction (punctuation handling, truncation limits) for production use. |
| **Bi-Encoder Training** | 🟢 Implemented | InfoNCE contrastive loop is fully functional. |
| **Cross-Encoder Reranker** | 🔴 Planned | Architecture defined in configs, but code is not yet implemented. |
| **IR Evaluation** | 🟡 Demo | Metrics (Recall@k, MRR, nDCG) are mathematically correct and unit-tested. **Note:** CLI currently requires manual generation of `predictions.json`. |
| **End-to-End Generation** | 🔴 Demo | `generate` command uses simulated contexts. **Planned:** Integration with the actual Bi-Encoder/Reranker pipeline. |
| **Faithfulness Evaluator** | 🔴 Demo | Uses a simple lexical overlap proxy. **Planned:** Integration of an NLI-based entailment model. |

---

## 🛠️ Installation

This framework is built as a standard, pip-installable Python package.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Aly-sher/Urdu-rag-framework.git
   cd Urdu-rag-framework

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate

 3. **Install dependencies:**
    ```bash
    pip install -e .
    pip install -e ".[dev]"

---

## 🚀 Usage & CLI

The framework exposes a Command Line Interface (CLI) via the urdu-rag command.

Note: As this is a research prototype, some commands (like train-cpt) are currently configured as "smoke tests" (e.g., 10 steps) to verify the pipeline logic without requiring massive compute resources.

# Phase 1: Data & Typology
```bash
urdu-rag fetch-data --samples 500
urdu-rag align-typology

# Phase 2: Tokenization & Adaptation
```bash
urdu-rag train-tokenizer
urdu-rag setup-lora
urdu-rag train-cpt

# Phase 3: Retrieval & Evaluation
```bash
urdu-rag mine-hard-negatives
urdu-rag train-retriever
urdu-rag evaluate-retrieval
---


## 🧪 Testing
The framework includes a suite of unit tests to verify the mathematical correctness of the evaluation metrics and data processing logic.
Run the test suite with:
```bash
pytest tests/ -v

## Current Coverage:
- ✅ test_evaluation.py: Validates Recall@k, MRR, and nDCG@k calculations.
- ✅ test_hard_negatives.py: Validates BM25 mining logic and triplet generation.
---

## 🗺️ Roadmap (Track B)
The following items are identified as critical next steps for moving this framework from a prototype to a publication-grade system:

- Real WECHSEL Initialization: Implement token-level alignment between Hindi and Urdu embeddings to replace the current distance-only typology step.
- Validated Ablation Study: distinct training runs comparing Typology-Init vs. Random-Init with proper statistical significance testing.
- NLI-Based Faithfulness: Replace the lexical overlap proxy with a model-based Natural Language Inference (NLI) check for hallucination detection.
- Full-Scale Benchmarking: Evaluate against standard datasets (e.g., XQuAD-ur) rather than the current 5-query proof-of-concept.
- Cross-Encoder Integration: Implement the Stage-2 reranking module.
---




  
   
