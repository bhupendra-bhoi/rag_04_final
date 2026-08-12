# Real World RAG System

A Retrieval-Augmented Generation (RAG) system built as a capstone project by **Group 18**, IIIT Hyderabad AIML Post-Graduate Certificate Program.

## Overview

This project implements a real-world RAG system with an interactive web interface that allows users to explore different domains, ask questions, and evaluate the quality of generated answers using TRACe metrics. The system is deployed as a Hugging Face Space: [real-world-rag](https://huggingface.co/spaces/HarshitaNalajala/real-world-rag)

<img width="1877" height="1079" alt="image" src="https://github.com/user-attachments/assets/bf8a5825-2ad5-48e8-93af-357c4fdae3eb" />


## Features

### Interactive Dashboard
- **Domain Selection**: Choose from various domains (e.g., Biomedical Research)
- **Question Selection**: Pick from curated questions within each domain
- **Pipeline Controls**: Customize the RAG pipeline with multiple options:
  - Generator model selection (e.g., llama-3.3-70b-versatile)
  - Hybrid retrieval (BM25 + dense retrieval)
  - Passage selection (top-k slider)
  - Cross-encoder reranking
  - Temperature control
  - Prompt templates (Biomedical or Generic)

### Evaluation Metrics
The system provides comprehensive evaluation using TRACe metrics:
- **Context Relevance**: How relevant the retrieved context is to the question
- **Utilization**: How well the generated answer uses the retrieved context
- **Completeness**: How complete the answer is relative to the ground truth
- **Adherence**: How well the answer adheres to the prompt requirements
- **Recall**: Traditional retrieval recall metric

### Real-time Analysis
- Get answers and metrics with a single click
- Compare different pipeline configurations
- Visualize how parameter changes affect performance


## Project Structure

```
rag_04_final/
├── real-world-rag/          # Main RAG system with Gradio interface
│   ├── app.py              # Gradio application
│   ├── deploy_artifacts/   # Deployed model and index files
│   └── requirements.txt    # Python dependencies
├── All_Domains_Combined_Notebooks/  # Domain-specific experimental notebooks
└── Experiments/            # Comprehensive experimental framework
```

## Experiments and Notebooks

### All_Domains_Combined_Notebooks/
This directory contains Jupyter notebooks and Python scripts for experiments conducted across different domains:

- **Biomedical**: `biomedical_RAGBench.ipynb` - RAG experiments on biomedical data using RAGBench dataset
- **Customer Support**: `customer_support_production_pipeline_colab.ipynb` - Production pipeline experiments for customer support domain
- **Finance**: `finance_RAG_Evaluation.ipynb` - RAG evaluation experiments on financial data
- **General Knowledge**: `gk_AIML_RAG_Bench.ipynb` - RAG benchmarking on general knowledge/AIML questions
- **Legal**: `legal_RAG_cuad_v2.ipynb` - RAG experiments on legal documents using CUAD dataset

Each notebook includes domain-specific data processing, retrieval experiments, and evaluation metrics. Python scripts (`.py` files) accompany the notebooks for production-ready implementations.

### Experiments/
A comprehensive experimental framework with modular components for systematic RAG experimentation:

- **config/** - Configuration files for different experiment setups
- **core/** - Core utilities and base classes
- **data_sources/** - Data loading and preprocessing modules
- **embedding/** - Embedding model implementations and utilities
- **evaluation/** - Evaluation metrics and scoring functions
- **experiment_configs/** - Predefined experiment configurations
- **notebooks/** - Analysis and exploration notebooks
- **parsers/** - Document parsing utilities
- **providers/** - API provider integrations (OpenRouter, etc.)
- **rag/** - RAG pipeline implementations
- **rag-experiments/** - Specific RAG experimental setups
- **reporting/** - Result generation and reporting tools
- **scripts/** - Utility scripts for running experiments
- **vectorstore/** - Vector database implementations
- **production/** - Production-ready pipeline components

This modular structure allows for systematic experimentation across different retrieval strategies, embedding models, and evaluation methodologies.

## Getting Started

### Prerequisites
- Python 3.8+
- Groq API key (for answer generation)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd rag_04_final
```

2. Install dependencies:
```bash
cd real-world-rag
pip install -r requirements.txt
```

3. Set up your Groq API key:
```bash
export GROQ_API_KEY=your_api_key_here
```

### Running the Application

```bash
cd real-world-rag
python app.py
```

The application will start locally and can be accessed through the provided URL.

## Deployment

The system is deployed on Hugging Face Spaces. To deploy your own version:

1. Push the `real-world-rag` directory to a Hugging Face Space
2. Add your Groq API key in Space Settings → Variables and secrets
3. Restart the Space

## Dataset

The system is evaluated on the biomedical (CovidQA) subset of [RAGBench](https://huggingface.co/datasets/rungalileo/ragbench).

## Team

- **Group 18 | RAG - 04**, IIIT Hyderabad AIML PGCP
- Supervisor: Dr. Manish Shrivastava
- Mentors: Gopichand, Lokesh

## Acknowledgments

- Built as part of the IIIT Hyderabad AIML Post-Graduate Certificate Program capstone project
- Uses models and datasets from Hugging Face
- Powered by Groq for fast inference

## Disclaimer

This is a research demo built on a fixed benchmark corpus. It is not a source of medical, financial, customer support, general knowledge or legal advice and should not be used for clinical, financial, customer support or legal decision-making.
