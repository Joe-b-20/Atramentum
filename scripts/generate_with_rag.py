# scripts/generate_with_rag.py
#!/usr/bin/env python3
"""
Generate journal entries with RAG-enhanced memory. Remembering to forget.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.atramentum.inference.generator import JournalGenerator
from src.atramentum.utils import io as aio
from src.atramentum.utils import logging as alog


class RAGGenerator:
    """Generate with retrieval-augmented memory."""
    
    def __init__(
        self,
        model_name: str,
        adapter_path: str = None,
        index_dir: str = "index/faiss/bge_small"
    ):
        self.logger = alog.get_logger(__name__)
        
        # Load generator
        self.generator = JournalGenerator(model_name, adapter_path)
        
        # Load encoder for retrieval
        self.encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")
        
        # Load FAISS index
        index_path = Path(index_dir) / "index.faiss"
        meta_path = Path(index_dir) / "meta.jsonl"
        
        self.index = faiss.read_index(str(index_path))
        self.metadata = aio.read_jsonl(meta_path)
        
        self.logger.info(f"Loaded index with {self.index.ntotal} vectors")
    
    def retrieve_memories(self, query: str, k: int = 4) -> str:
        """Retrieve relevant memory snippets."""
        # Encode query
        query_embedding = self.encoder.encode(
            [query],
            normalize_embeddings=True
        ).astype('float32')
        
        # Search
        distances, indices = self.index.search(query_embedding, k)
        
        # Build memory block
        memories = []
        for idx in indices[0]:
            if idx < len(self.metadata):
                item = self.metadata[idx]
                # Extract first 2 sentences as snippet
                text = item['text']
                sentences = text.split('. ')[:2]
                snippet = '. '.join(sentences)
                
                memory = f"[{item['date']}] {snippet}"
                memories.append(memory)
        
        return '\n'.join(memories)
    
    def generate(
        self,
        prompt: str,
        mode: str = "generate",
        k: int = 4,
        max_new_tokens: int = 800,
        use_rag: bool = True
    ) -> str:
        """Generate with optional RAG enhancement."""
        
        # Retrieve memories if enabled
        memory = ""
        if use_rag:
            self.logger.info(f"Retrieving {k} memory snippets")
            memory = self.retrieve_memories(prompt, k)
            
            if memory:
                self.logger.info("Memory block created:")
                print(f"\n--- MEMORY ---\n{memory}\n--------------\n")
        
        # Generate
        self.logger.info("Generating entry...")
        result = self.generator.generate(
            prompt=prompt,
            memory=memory,
            mode=mode,
            max_new_tokens=max_new_tokens,
            temperature=0.8,
            top_p=0.95
        )
        
        # Ensure date-first format
        if not result.strip().startswith(datetime.now().strftime('%m/%d/%Y')):
            today = datetime.now().strftime('%m/%d/%Y')
            result = f"{today} —\n{result}"
        
        return result


def main():
    parser = argparse.ArgumentParser(
        description="Generate with RAG. Memory, unreliable but present."
    )
    parser.add_argument('--model', type=str, default='meta-llama/Meta-Llama-3.1-8B-Instruct')
    parser.add_argument('--adapter', type=str, default=None)
    parser.add_argument('--index-dir', type=str, default='index/faiss/bge_small')
    parser.add_argument('--seed', type=str, default=None, help='Seed text to continue')
    parser.add_argument('--rewrite', type=str, default=None, help='Text to rewrite')
    parser.add_argument('--prompt', type=str, default=None, help='Generation prompt')
    parser.add_argument('--k', type=int, default=4, help='Number of memories to retrieve')
    parser.add_argument('--max-new', type=int, default=800)
    parser.add_argument('--no-rag', action='store_true', help='Disable RAG')
    
    args = parser.parse_args()
    
    # Determine mode and content
    if args.seed:
        mode = 'seed'
        content = args.seed
    elif args.rewrite:
        mode = 'rewrite'
        content = args.rewrite
    else:
        mode = 'generate'
        content = args.prompt or "Write about the exhaustion of modern existence"
    
    # Generate
    generator = RAGGenerator(args.model, args.adapter, args.index_dir)
    result = generator.generate(
        prompt=content,
        mode=mode,
        k=args.k,
        max_new_tokens=args.max_new,
        use_rag=not args.no_rag
    )
    
    print("\n--- GENERATED ENTRY ---")
    print(result)
    print("-----------------------")


if __name__ == '__main__':
    main()