# 📚 Class IX History RAG Assistant

A Retrieval-Augmented Generation (RAG) chatbot that answers questions from the NCERT Class IX History textbook, *India and the Contemporary World – I*, grounding every answer strictly in the source material.

**🔗 Live app:** [rag-history-assistant-kv5xtxxe7dfjygvbbwwc4s.streamlit.app](https://rag-history-assistant-kv5xtxxe7dfjygvbbwwc4s.streamlit.app/)

---

## Overview

Instead of relying on a language model's parametric (and often unreliable) knowledge, this assistant retrieves the most relevant passages from the actual textbook before generating a response. This keeps answers grounded in the source text and reduces hallucination — a core motivation behind the RAG approach.

**Pipeline:**

```
Question → Embedding → FAISS Retrieval → CrossEncoder Reranking → Qwen2.5 → Answer
```

1. **Embed** — the user's question is embedded using a Sentence Transformer.
2. **Retrieve** — the top-k most similar passages are pulled from a FAISS vector index built over the textbook.
3. **Rerank** — a CrossEncoder rescoring model reorders the retrieved passages by true relevance to the query, and only the top 2 are kept as context.
4. **Generate** — the selected context and question are passed to Qwen2.5-0.5B-Instruct, which is prompted to answer using *only* the supplied context.

> **Note:** The generation model is a compact 0.5B-parameter LLM chosen for fast, low-cost inference on free-tier hosting. Because of its size, occasional factual or contextual errors are possible — answers should be verified against the textbook when accuracy matters.

## Features

- 🔎 Semantic search over the full NCERT Class IX History (Book I) textbook
- 🎯 Two-stage retrieval (FAISS + CrossEncoder reranking) for higher-precision context selection
- 🤖 Context-grounded generation with an instruction-tuned LLM
- 💬 Persistent chat interface with conversation history and a "Clear Chat" control
- ⚡ Lightweight stack designed to run on free-tier Streamlit Cloud hosting

## Tech Stack

| Component | Technology |
|---|---|
| UI / App framework | [Streamlit](https://streamlit.io/) |
| Embedding model | `all-MiniLM-L6-v2` (Sentence Transformers) |
| Vector index | [FAISS](https://github.com/facebookresearch/faiss) |
| Reranker | `cross-encoder/ms-marco-electra-base` |
| Generation model | [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) |
| ML framework | PyTorch / Hugging Face Transformers |

## Project Structure

```
RAG-History-Assistant/
├── app/
│   ├── model.py         # Loads and caches the embedding, reranker, and LLM models
│   └── inference.py      # Retrieval → reranking → generation pipeline
├── data/
│   ├── NCERT_CLASS_9_HISTORY_INDIA_AND_CONTEMPORARY_WORLD_1.pdf  # Source textbook
│   ├── docs.pkl           # Preprocessed/chunked text passages
│   └── index.faiss        # Prebuilt FAISS vector index over the passages
├── streamlit_app.py       # App entry point (chat UI)
├── requirements.txt
└── LICENSE
```

## Getting Started

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
git clone https://github.com/Sourith2008/RAG-History-Assistant.git
cd RAG-History-Assistant
pip install -r requirements.txt
```

### Run locally

```bash
streamlit run streamlit_app.py
```

The app will be available at `http://localhost:8501`. On first run, the embedding, reranking, and generation models will be downloaded from Hugging Face and cached locally.

## How It Works

- **`app/model.py`** loads and `@st.cache_resource`-caches all four models (LLM, tokenizer, embedder, reranker) so they're only initialized once per session.
- **`app/inference.py`** loads the prebuilt FAISS index and document store (`data/index.faiss`, `data/docs.pkl`), and exposes an `infer(user_input)` function that runs the full retrieve → rerank → generate pipeline and returns a grounded answer.
- **`streamlit_app.py`** wires this up to a chat interface, maintaining conversation state in `st.session_state`.

The system prompt constrains the model to answer only from the supplied context, avoid outside knowledge, avoid conflating pre- and post-French Revolution events, and follow the user's requested answer format (e.g., one-word answers, numbered points).

## Limitations

- Answers are only as good as the retrieved context — ambiguous or out-of-scope questions may return incomplete answers.
- The 0.5B generation model can occasionally misinterpret retrieved context or produce imprecise phrasing.
- Currently scoped to a single textbook (*India and the Contemporary World – I*); it will not answer questions outside this source.

## License

This project is licensed under the [Apache License 2.0](LICENSE).

## Acknowledgements

- [NCERT](https://ncert.nic.in/) for the source textbook, *India and the Contemporary World – I*
- [Qwen2.5](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) by Alibaba Cloud
- [Sentence Transformers](https://www.sbert.net/) and [FAISS](https://github.com/facebookresearch/faiss) for retrieval
