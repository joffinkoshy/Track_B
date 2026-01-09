"""
Narrative Processing Module

Handles loading, preprocessing, and segmentation of long narrative texts.
"""

import re
from typing import Dict, List, Optional
from dataclasses import dataclass
import numpy as np

@dataclass
class NarrativeSegment:
    """Represents a segment of narrative text with positional information"""
    text: str
    start_pos: int
    end_pos: int
    segment_id: str
    features: Dict[str, float]
    raw_text: Optional[str] = None

class NarrativeProcessor:
    """Processes long narratives into segments with feature extraction"""

    def __init__(self, segment_size: int = 1000, overlap: int = 100):
        """
        Initialize narrative processor

        Args:
            segment_size: Size of each narrative segment in characters
            overlap: Overlap between consecutive segments
        """
        self.segment_size = segment_size
        self.overlap = overlap
        self.vocab = set()
        self.character_names = set()
        self.segment_count = 0

    def preprocess_text(self, text: str, preserve_case: bool = False) -> str:
        """Basic text preprocessing with option to preserve case"""
        if not preserve_case:
            text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text.strip()

    def segment_narrative(self, text: str, preserve_case: bool = False) -> List[NarrativeSegment]:
        """Segment long narrative into manageable chunks with overlap"""
        segments = []
        text_length = len(text)
        self.segment_count = 0

        # Calculate step size considering overlap
        step = self.segment_size - self.overlap

        for i in range(0, text_length, step):
            end = min(i + self.segment_size, text_length)
            segment_text = text[i:end]
            raw_segment = text[i:end]  # Preserve original for raw_text

            # Preprocess the segment text
            processed_text = self.preprocess_text(segment_text, preserve_case)

            segment = NarrativeSegment(
                text=processed_text,
                raw_text=raw_segment,
                start_pos=i,
                end_pos=end,
                segment_id=f"seg_{self.segment_count}",
                features=self._extract_segment_features(processed_text, raw_segment)
            )

            segments.append(segment)
            self.segment_count += 1

        return segments

    def _extract_segment_features(self, processed_text: str, raw_text: str) -> Dict[str, float]:
        """Extract comprehensive features from text segment"""
        words = processed_text.split() if processed_text else []
        raw_words = raw_text.split() if raw_text else []

        # Extract character names (simple heuristic: capitalized words in raw text)
        potential_names = [word for word in raw_words if word[0].isupper() and len(word) > 2]
        self.character_names.update(potential_names)

        return {
            'length': len(words),
            'raw_length': len(raw_words),
            'unique_words': len(set(words)),
            'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
            'sentence_count': processed_text.count('.') + processed_text.count('!') + processed_text.count('?'),
            'paragraph_count': processed_text.count('\n\n') + 1 if '\n\n' in processed_text else 1,
            'character_names': len(potential_names),
            'punctuation_density': sum(1 for char in raw_text if char in '.,;:!?') / len(raw_text) if raw_text else 0
        }

    def get_processing_stats(self) -> dict:
        """Get statistics about the processing"""
        return {
            'segments_created': self.segment_count,
            'unique_characters': len(self.character_names),
            'vocab_size': len(self.vocab)
        }

    def reset_stats(self) -> None:
        """Reset processing statistics"""
        self.segment_count = 0
        self.vocab = set()
        self.character_names = set()
