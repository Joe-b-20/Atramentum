# scripts/build_index.py
#!/usr/bin/env python3
"""
Build FAISS index for RAG. Creating a searchable void.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from src.atramentum.utils import io as aio
from src.atramentum.utils import logging as alog


class IndexBuilder:
    """Builds FAISS index for memory retrieval."""
    
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.logger = alog.get_logger(__name__)
        self.encoder = SentenceTransformer(model_name)
        
    def build_index(self, input_path: str = "data/processed/atra_labels.jsonl"):
        """Build FAISS index from labeled chunks."""
        # Load data
        data = aio.read_jsonl(input_path)
        self.logger.info(f"Loaded {len(data)} chunks")
        
        # Extract texts
        texts = [item['text'] for item in data]
        
        # Generate embeddings
        self.logger.info("Generating embeddings. Vectorizing the void...")
        embeddings = self.encoder.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True
        )
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product for normalized vectors
        
        # Add embeddings
        index.add(embeddings.astype('float32'))
        self.logger.info(f"Index built with {index.ntotal} vectors")
        
        # Save index and metadata
        output_dir = Path("index/faiss/bge_small")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_path = output_dir / "index.faiss"
        faiss.write_index(index, str(index_path))
        self.logger.info(f"Saved index to {index_path}")
        
        # Save metadata (everything except embeddings)
        metadata = []
        for item in data:
            meta_item = {k: v for k, v in item.items() if k != 'text'}
            meta_item['text'] = item['text']  # Keep text for retrieval
            metadata.append(meta_item)
        
        meta_path = output_dir / "meta.jsonl"
        aio.write_jsonl(meta_path, metadata)
        self.logger.info(f"Saved metadata to {meta_path}")
        
        return index_path, meta_path


def main():
    """Build the index."""
    builder = IndexBuilder()
    index_path, meta_path = builder.build_index()
    print(f"Index built: {index_path}")
    print(f"Metadata saved: {meta_path}")
    print("RAG index ready. Memories searchable, if not reliable.")


if __name__ == '__main__':
    main()