# Urdu-RAG Framework
### A Typology-Driven Retrieval-Augmented Generation Pipeline for Low-Resource Languages

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-Pending-b31b1b.svg)]()

> **Abstract:** While recent monolingual Urdu Large Language Models (e.g., Qalb-8B, Alif-1.0) have achieved state-of-the-art performance in generative tasks, they remain highly susceptible to factual hallucinations in specialized, localized domains due to reliance on parametric memory. This repository introduces a rigorous, end-to-end Retrieval-Augmented Generation (RAG) framework specifically architected for Urdu. By integrating strict $n$-gram data decontamination, typological proximity ranking, parameter-efficient Continual Pre-Training (CPT), and a two-stage cascade retrieval system (BM25 Hard-Negative Mining $\rightarrow$ Cross-Encoder Reranking), this framework provides a mathematically grounded, reproducible pipeline for low-resource RAG.

---

## 🏛️ Architecture & Key Contributions

Unlike standard RAG implementations that rely on off-the-shelf English embedding models, this framework is built from the ground up to address the specific morphological and semantic challenges of the Urdu language:

1. **Strict Data Hygiene & Decontamination:** Implements a pure-Python Unicode heuristic filter to eliminate code-switched noise, combined with strict $n$-gram overlap decontamination to prevent evaluation data leakage during Continual Pre-Training (CPT).
2. **Typological Proximity Ranking:** Utilizes the URIEL+ database to compute geometric distances between language feature vectors, identifying the optimal donor language (Hindi) for cross-lingual embedding initialization.
3. **Parameter-Efficient Adaptation:** Employs Low-Rank Adaptation (LoRA) targeting both attention and MLP projection matrices, restricting trainable parameters to $<1\%$ of the base model to prevent catastrophic forgetting of foundational reasoning capabilities.
4. **Cascade Contrastive Retrieval:** 
   - *Stage 1:* A BM25 index mines "Hard Negatives" (documents with high lexical overlap but low semantic relevance) to train a Bi-Encoder using InfoNCE loss, forcing deep Urdu semantic understanding.
   - *Stage 2:* A Cross-Encoder reranker processes the top-$k$ candidates via deep cross-attention to maximize retrieval precision before LLM injection.
5. **Standard IR Evaluation Harness:** Built-in computation of industry-standard Information Retrieval metrics, including Recall@$k$, Mean Reciprocal Rank (MRR), and Normalized Discounted Cumulative Gain (nDCG@$k$).

---

## 🛠️ Installation

This framework is built as a standard, pip-installable Python package. 

1. Clone the repository:
   ```bash
   git clone https://github.com/Aly-sher/Urdu-rag-framework.git
   cd Urdu-rag-framework

Create and activate a virtual environment (Python 3.10+ recommended):
python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate

Install the framework and all dependencies in editable mode:
 pip install -e .

## 🚀 Reproducing the Pipeline
The framework exposes a professional Command Line Interface (CLI) via the urdu-rag command. You can execute the entire research pipeline sequentially:
## Phase 1: Data Fetching & Typological Alignment
Fetch and clean the Urdu Wikipedia/OSCAR corpus
urdu-rag fetch-data --samples 500

Compute URIEL+ typological distances and identify the closest anchor language
urdu-rag align-typology

## Phase 2: Tokenization & Parameter-Efficient Setup
Train a custom Byte-Level BPE tokenizer optimized for the Nastaliq script
urdu-rag train-tokenizer

Initialize the base model (e.g., Qwen2.5) and inject LoRA adapters
urdu-rag setup-lora

## Phase 3: Data Hygiene & Contrastive Retrieval
Mine BM25 hard negatives from the corpus for InfoNCE contrastive training
urdu-rag mine-hard-negatives

Evaluate the retrieval pipeline using standard IR metrics (Recall@k, MRR, nDCG)
urdu-rag evaluate-retrieval

## 📊 Evaluation Metrics

The framework includes a dedicated evaluation harness (src/urdu_rag/core/evaluation.py) designed to rigorously benchmark the retrieval stage against human-annotated Urdu query-document relevance sets.

Supported metrics include:

-Recall@k:Measures the fraction of relevant documents retrieved in the top k results.
-MRR (Mean Reciprocal Rank): Evaluates the rank of the first relevant document retrieved.
-nDCG@k(Normalized Discounted Cumulative Gain): Measures ranking quality by assigning higher relevance scores to documents appearing at the top of the list.

## 📖 Citation
If you use this framework, pipeline, or evaluation harness in your research, please cite our upcoming paper:
@misc{urdu_rag_framework_2026,
  author = Ali Sher Khan Tareen,
  title = {A Typology-Driven Retrieval-Augmented Generation Framework for Low-Resource Languages},
  year = {2026},
  publisher = https://github.com/Aly-sher,
  journal = https://github.com/Aly-sher/Urdu-rag-framework,
  howpublished = {\url{https://github.com/Aly-sher/Urdu-rag-framework}},
}

## ⚖️ License
This project is licensed under the MIT License - see the LICENSE file for details.