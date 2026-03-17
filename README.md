# Mini RAG Assistant

A lightweight Retrieval-Augmented Generation (RAG) Question Answering system built with semantic retrieval, reranking, and LLM-based answer generation.

This project demonstrates a full RAG pipeline including document retrieval, prompt construction, LLM generation, evaluation, and an interactive CLI demo.

---

# System Architecture

User Question
↓
Embedding Retrieval (Sentence Transformers + FAISS)
↓
Rerank
↓
Top Context Selection
↓
Prompt Builder
↓
LLM Generation
↓
Final Answer + Evidence

---

# Features

Semantic Retrieval

* Sentence-Transformers embeddings
* FAISS vector index

Two-stage Retrieval

* Top-k recall
* Rerank for improved relevance

Prompt Engineering

* Structured prompt with strict answer constraints
* Refuse-to-answer mechanism when evidence is insufficient

LLM Generation

* Integrated with local LLM via Ollama

Evaluation Pipeline

* Batch QA evaluation
* Exact Match
* Keyword Hit Score
* Retrieval Hit

Interactive CLI Demo

* Chat interface for testing RAG QA

---

# Project Structure

```
mini-rag-assistant
│
├── chat.py
├── requirements.txt
│
├── data
│   ├── documents.py
│   └── qa_test.json
│
├── outputs
│   ├── predictions.json
│   └── eval_report.txt
│
└── src
    ├── retriever.py
    ├── generator.py
    ├── prompt_builder.py
    ├── rag_pipeline.py
    └── evaluate.py
```

---

# Installation

Clone the repository

```
git clone https://github.com/yourname/mini-rag-assistant.git
cd mini-rag-assistant
```

Install dependencies

```
pip install -r requirements.txt
```

Install Ollama (for local LLM)

https://ollama.com

Pull a model

```
ollama pull qwen2:7b
```

---

# Run Interactive Demo

```
python chat.py
```

Example

```
User: 猫是什么动物？

AI: 猫是哺乳动物，属于猫科动物。

Evidence
1. 猫是哺乳动物
2. 猫属于猫科动物
```

---

# Run Evaluation

```
python src/evaluate.py
```

Example metrics

```
Exact Match: 0.50
Keyword Hit: 0.81
Retrieval Hit: 0.88
```

---

# Key Components

retriever.py
Semantic retrieval using Sentence-Transformers and FAISS.

generator.py
LLM-based answer generation using Ollama API.

prompt_builder.py
Constructs structured prompts for RAG generation.

rag_pipeline.py
Orchestrates the full RAG pipeline.

evaluate.py
Batch evaluation for QA dataset.

---

# Future Improvements

* Cross-Encoder reranking
* Multi-document reasoning
* Web UI (Streamlit / Gradio)
* Hybrid retrieval (BM25 + embedding)
* Multimodal RAG

---

# Author

Built as a learning project for RAG systems and LLM-based QA pipelines.
