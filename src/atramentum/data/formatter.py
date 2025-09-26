# src/atramentum/data/formatter.py
"""
Data formatter - turning raw journal chaos into structured despair.
"""

import re
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import yaml

from ..utils import io as aio
from . import validators


class JournalFormatter:
    """Formats journal entries for training. Handles dates like a therapist handles trauma: carefully."""
    
    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        # Compile the date regex once, suffer forever
        self.date_pattern = re.compile(self.config['date_regex'])
        self.min_chars = self.config.get('min_chars', 50)
        self.max_chars = self.config.get('max_chars', 4000)
        self.seeds_ratio = self.config.get('seeds_ratio', 0.3)
        
    def parse_entries(self, text: str) -> List[Dict]:
        """Parse text into entries by date lines. Each date starts a new existential crisis."""
        entries = []
        current_entry = None
        current_date = None
        
        for line in text.split('\n'):
            # Check if line is a date
            if self.date_pattern.match(line.strip()):
                # Save previous entry if exists
                if current_entry and len(current_entry['text']) >= self.min_chars:
                    entries.append(current_entry)
                
                # Start new entry
                date_str = line.strip()
                current_date = self._normalize_date(date_str)
                current_entry = {
                    'date': current_date,
                    'text': '',
                    'raw_date': date_str
                }
            elif current_entry is not None:
                # Add to current entry
                current_entry['text'] += line + '\n'
        
        # Don't forget the last entry (like we forget everything else)
        if current_entry and len(current_entry['text']) >= self.min_chars:
            entries.append(current_entry)
            
        return entries
    
    def _normalize_date(self, date_str: str) -> str:
        """Normalize date to MM/DD/YYYY format. Time is a flat circle anyway."""
        # Remove whitespace
        date_str = date_str.strip()
        
        # Parse various formats (mm/dd/yyyy, m/d/yy, etc)
        for fmt in ['%m/%d/%Y', '%m/%d/%y', '%-m/%-d/%Y', '%-m/%-d/%y']:
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt.strftime('%m/%d/%Y')
            except ValueError:
                continue
        
        # If all else fails, return as-is (YOLO)
        return date_str
    
    def create_sft_dataset(self, entries: List[Dict]) -> List[Dict]:
        """Create SFT training examples. Teaching models to write like us at 3 AM."""
        dataset = []
        
        for entry in entries:
            # Clean up the text
            text = self._clean_text(entry['text'])
            if not text:
                continue
            
            # Create both rewrite and seed examples
            year = entry['date'].split('/')[-1]
            formatted_text = f"{entry['date']} —\n[YEAR: {year}]\n{text}"
            
            # Calculate hash for dedup
            entry_hash = hashlib.md5(f"{entry['date']}{text[:64]}".encode()).hexdigest()
            
            # Rewrite example (70%)
            if len(dataset) % 10 < 7:  # Simple ratio enforcement
                dataset.append({
                    'messages': [
                        {'role': 'system', 'content': 'You are Atramentum, a journal assistant.'},
                        {'role': 'user', 'content': f"Rewrite this entry in journal style:\n{text[:200]}"},
                        {'role': 'assistant', 'content': formatted_text}
                    ],
                    'meta': {
                        'type': 'rewrite',
                        'date': entry['date'],
                        'hash': entry_hash,
                        'source': 'journal'
                    }
                })
            else:
                # Seed example (30%)
                seed = text[:50] + "..."
                dataset.append({
                    'messages': [
                        {'role': 'system', 'content': 'You are Atramentum, a journal assistant.'},
                        {'role': 'user', 'content': f"Continue this journal entry from {entry['date']}:\n{seed}"},
                        {'role': 'assistant', 'content': formatted_text}
                    ],
                    'meta': {
                        'type': 'seed',
                        'date': entry['date'],
                        'hash': entry_hash,
                        'source': 'journal'
                    }
                })
        
        return dataset
    
    def _clean_text(self, text: str) -> str:
        """Clean text while preserving the essential darkness."""
        # Remove excessive whitespace
        text = re.sub(r'\n\n+', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        
        # Strip quotes if configured
        if self.config.get('strip_quotes', True):
            text = text.strip('"\'""''')
        
        return text.strip()
    
    def process_files(self) -> str:
        """Process all input files and save the dataset. Returns output path or existential void."""
        all_entries = []
        
        for input_path in self.config['input_paths']:
            path = Path(input_path)
            if not path.exists():
                print(f"Warning: {input_path} not found. Like meaning in life.")
                continue
            
            # Read file based on extension
            if path.suffix == '.txt':
                text = path.read_text(encoding='utf-8')
            elif path.suffix == '.docx':
                import docx
                doc = docx.Document(path)
                text = '\n'.join([para.text for para in doc.paragraphs])
            else:
                print(f"Unsupported format: {path.suffix}")
                continue
            
            entries = self.parse_entries(text)
            all_entries.extend(entries)
        
        # Create dataset
        dataset = self.create_sft_dataset(all_entries)
        
        # Save to JSONL
        output_dir = Path(self.config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / 'atra_sft.jsonl'
        
        aio.write_jsonl(output_path, dataset)
        print(f"Saved {len(dataset)} examples to {output_path}")
        
        return str(output_path)