# scripts/auto_label.py
#!/usr/bin/env python3
"""
Auto-label journal entries for RAG. Categorizing the chaos.
"""

import sys
import hashlib
from pathlib import Path
from typing import List, Dict
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from transformers import AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

from src.atramentum.utils import io as aio
from src.atramentum.utils import metrics
from src.atramentum.utils import logging as alog


class EntryLabeler:
    """Labels and chunks entries for RAG indexing."""
    
    def __init__(self, chunk_size: int = 900, overlap: int = 120):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.logger = alog.get_logger(__name__)
        
        # Load tokenizer for chunking
        self.tokenizer = AutoTokenizer.from_pretrained(
            "BAAI/bge-small-en-v1.5",
            trust_remote_code=True
        )
    
    def chunk_text(self, text: str) -> List[Dict]:
        """Chunk text with overlap. Breaking down breakdowns."""
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        chunks = []
        
        # Calculate chunk positions
        start = 0
        chunk_id = 0
        total_chunks = (len(tokens) - 1) // (self.chunk_size - self.overlap) + 1
        
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            
            # Decode chunk
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            
            # Determine position
            if chunk_id == 0:
                position = "intro"
            elif chunk_id == total_chunks - 1:
                position = "outro"
            else:
                position = "middle"
            
            chunks.append({
                'text': chunk_text,
                'chunk_id': chunk_id,
                'n_chunks': total_chunks,
                'position': position
            })
            
            # Move forward with overlap
            start = end - self.overlap if end < len(tokens) else end
            chunk_id += 1
        
        return chunks
    
    def extract_topics(self, texts: List[str], max_topics: int = 5) -> List[List[str]]:
        """Extract topics via TF-IDF. Finding themes in the chaos."""
        if not texts:
            return []
        
        # Fit TF-IDF
        vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # Extract top topics per document
            all_topics = []
            for doc_idx in range(tfidf_matrix.shape[0]):
                doc_tfidf = tfidf_matrix[doc_idx].toarray().flatten()
                top_indices = np.argsort(doc_tfidf)[-max_topics:][::-1]
                topics = [feature_names[i] for i in top_indices if doc_tfidf[i] > 0]
                all_topics.append(topics)
            
            return all_topics
        except:
            # Fallback for edge cases
            return [[] for _ in texts]
    
    def process_dataset(self, input_path: str = "data/processed/atra_sft.jsonl"):
        """Process SFT dataset to create labeled chunks."""
        data = aio.read_jsonl(input_path)
        
        # Extract unique entries
        seen = set()
        unique_entries = []
        
        for item in data:
            # Get assistant text and date
            assistant_msg = item['messages'][2]['content']
            date = item['meta']['date']
            
            # Create unique key
            key = hashlib.md5(f"{date}{assistant_msg[:64]}".encode()).hexdigest()
            
            if key not in seen:
                seen.add(key)
                unique_entries.append({
                    'date': date,
                    'text': assistant_msg,
                    'meta': item['meta']
                })
        
        self.logger.info(f"Processing {len(unique_entries)} unique entries")
        
        # Process each entry
        labeled_data = []
        all_texts = []
        
        for entry in unique_entries:
            # Parse date for year
            date_obj = datetime.strptime(entry['date'], '%m/%d/%Y')
            year = date_obj.year
            
            # Chunk the text
            chunks = self.chunk_text(entry['text'])
            
            for chunk in chunks:
                # Calculate metrics
                chunk_metrics = metrics.compute_entry_metrics(chunk['text'])
                
                labeled_item = {
                    'id': hashlib.md5(f"{entry['date']}{chunk['chunk_id']}".encode()).hexdigest(),
                    'date': entry['date'],
                    'year': year,
                    'chunk_id': chunk['chunk_id'],
                    'n_chunks': chunk['n_chunks'],
                    'position': chunk['position'],
                    'text': chunk['text'],
                    'metrics': chunk_metrics,
                    'source': entry['meta'].get('source', 'journal')
                }
                
                labeled_data.append(labeled_item)
                all_texts.append(chunk['text'])
        
        # Extract topics across all chunks
        self.logger.info("Extracting topics via TF-IDF")
        all_topics = self.extract_topics(all_texts)
        
        # Add topics to labeled data
        for item, topics in zip(labeled_data, all_topics):
            item['topics'] = topics
            
            # Extract mood from metrics
            mood_scores = item['metrics']['mood_indicators']
            item['mood'] = max(mood_scores.items(), key=lambda x: x[1])[0] if mood_scores else 'neutral'
            
            # Extract motifs from metrics
            item['motifs'] = item['metrics']['motifs']
        
        # Save labeled dataset
        output_path = Path("data/processed/atra_labels.jsonl")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        aio.write_jsonl(output_path, labeled_data)
        self.logger.info(f"Saved {len(labeled_data)} labeled chunks to {output_path}")
        
        return output_path


def main():
    """Run the labeling process."""
    labeler = EntryLabeler()
    labeler.process_dataset()
    print("Labeling complete. Order imposed on chaos.")


if __name__ == '__main__':
    main()