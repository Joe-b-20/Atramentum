# Atramentum

*A self-hosted journal model that remembers like you do: poorly, but with conviction.*

## Overview

Atramentum is a QLoRA-trained language model specialized in replicating personal journal entries with a date-first format, darkly sarcastic voice, and "archive memory" via local RAG.


## Architecture
```mermaid
flowchart TD
    A["Raw Journal Export"] -- Parse by Date --> B["Entry Chunks"]
    B -- 70% Rewrites --> C["SFT Dataset"]
    B -- 30% Seeds --> C
    C -- "Recency-Weighted" --> D["QLoRA Training"]
    D -- Optional --> E["DPO Hardening"]
    B -- "TF-IDF + Embeddings" --> F["FAISS Index"]
    F -- RAG Retrieval --> G["Memory Context"]
    H["User Prompt"] --> I["API Server"]
    I -- Query --> F
    G --> J["LLM Generation"]
    E --> J
    J -- "Date-First Entry" --> K["Output"]

    style A fill:#FFE0B2
    style K fill:#BBDEFB




## Quick Start

See [docs/quickstart.md](docs/quickstart.md) for detailed setup instructions.

## License

MIT © JB
