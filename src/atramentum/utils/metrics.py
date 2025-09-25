# src/atramentum/utils/metrics.py
"""
Metrics utilities. Quantifying the qualitative despair.
"""

import re
from typing import Dict, List
from collections import Counter


def calculate_dash_rate(text: str) -> float:
    """Calculate em-dash usage rate. The punctuation of interrupted thoughts."""
    dashes = text.count('—') + text.count('--')
    sentences = len(re.findall(r'[.!?]+', text)) or 1
    return dashes / sentences


def calculate_punchline_density(text: str) -> float:
    """Estimate punchline/joke density. Humor as coping mechanism."""
    # Look for common humor markers
    markers = [
        r'\([^)]+\)',  # Parenthetical asides
        r'\.\.\.',     # Ellipses (timing)
        r'[!?]{2,}',   # Multiple punctuation
        r'\b(anyway|apparently|somehow|totally|literally|actually)\b',
    ]
    
    count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in markers)
    words = len(text.split())
    
    return count / max(words, 1) * 100


def extract_mood_indicators(text: str) -> Dict[str, float]:
    """Extract mood indicators from text. The emotional weather report."""
    moods = {
        'despair': ['hopeless', 'meaningless', 'void', 'empty', 'nothing'],
        'anger': ['fuck', 'hate', 'rage', 'pissed', 'furious'],
        'anxiety': ['worry', 'anxious', 'panic', 'scared', 'nervous'],
        'tired': ['exhausted', 'tired', 'drained', 'sleep', 'insomnia'],
        'sardonic': ['whatever', 'apparently', 'somehow', 'obviously', 'clearly']
    }
    
    text_lower = text.lower()
    scores = {}
    
    for mood, keywords in moods.items():
        count = sum(text_lower.count(word) for word in keywords)
        scores[mood] = count / len(text.split()) * 100
    
    return scores


def extract_motifs(text: str) -> List[str]:
    """Extract recurring motifs. The themes we can't escape."""
    motif_patterns = {
        'family': r'\b(mother|father|parent|sibling|brother|sister|family)\b',
        'capitalism': r'\b(money|work|job|boss|capitalism|corporate|wage)\b',
        'body': r'\b(body|pain|sick|health|medical|doctor|tired)\b',
        'alcohol': r'\b(drink|drunk|alcohol|beer|wine|whiskey|bar)\b',
        'weather': r'\b(rain|sun|cloud|weather|storm|cold|hot|snow)\b',
        'memory': r'\b(remember|forget|memory|past|childhood|used to)\b',
        'time': r'\b(time|clock|hour|minute|day|night|morning|evening)\b',
        'death': r'\b(death|die|dead|mortality|funeral|grave)\b'
    }
    
    found_motifs = []
    for motif, pattern in motif_patterns.items():
        if re.search(pattern, text, re.IGNORECASE):
            found_motifs.append(motif)
    
    return found_motifs


def compute_entry_metrics(text: str) -> Dict:
    """Compute all metrics for a journal entry."""
    return {
        'dash_rate': calculate_dash_rate(text),
        'punchline_density': calculate_punchline_density(text),
        'mood_indicators': extract_mood_indicators(text),
        'motifs': extract_motifs(text),
        'word_count': len(text.split()),
        'char_count': len(text),
        'avg_sentence_length': len(text.split()) / max(len(re.findall(r'[.!?]+', text)), 1)
    }