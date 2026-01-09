"""
Representation Vectorizer Module

Handles conversion of text to numerical representations for BDH processing.
"""

import numpy as np
import hashlib
from typing import List, Dict, Optional
from collections import defaultdict

class TextVectorizer:
    """Converts text to numerical vector representations"""

    def __init__(self, vector_dim: int = 128, use_hashing: bool = True):
        """
        Initialize text vectorizer

        Args:
            vector_dim: Dimension of output vectors
            use_hashing: Whether to use hashing trick for vectorization
        """
        self.vector_dim = vector_dim
        self.use_hashing = use_hashing
        self.vocab = set()
        self.vocab_size = 0
        self.word_counts = defaultdict(int)

    def text_to_vector(self, text: str, method: str = 'hashing') -> np.ndarray:
        """
        Convert text to numerical vector

        Args:
            text: Input text to vectorize
            method: Vectorization method ('hashing', 'bow', 'tfidf')

        Returns:
            Numerical vector representation
        """
        if method == 'hashing':
            return self._hashing_vectorizer(text)
        elif method == 'bow':
            return self._bow_vectorizer(text)
        elif method == 'tfidf':
            return self._tfidf_vectorizer(text)
        else:
            return self._hashing_vectorizer(text)

    def _hashing_vectorizer(self, text: str) -> np.ndarray:
        """Convert text to vector using hashing trick"""
        vector = np.zeros(self.vector_dim)
        words = text.split()

        for i, word in enumerate(words[:self.vector_dim]):  # Limit to vector_dim words
            # Update vocabulary stats
            self.vocab.add(word)
            self.word_counts[word] += 1
            self.vocab_size = len(self.vocab)

            # Create hash-based representation
            word_hash = int(hashlib.md5(word.encode()).hexdigest(), 16) % 10000
            vector[i % self.vector_dim] = word_hash / 10000.0

        return vector

    def _bow_vectorizer(self, text: str) -> np.ndarray:
        """Convert text to bag-of-words vector"""
        words = text.split()
        vector = np.zeros(self.vector_dim)

        # Simple BOW: count word occurrences modulo vector_dim
        for word in words:
            self.vocab.add(word)
            self.word_counts[word] += 1
            self.vocab_size = len(self.vocab)

            # Use hash of word to determine position
            word_hash = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.vector_dim
            vector[word_hash] += 1

        # Normalize
        if len(words) > 0:
            vector = vector / len(words)

        return vector

    def _tfidf_vectorizer(self, text: str) -> np.ndarray:
        """Convert text to TF-IDF like vector (simplified)"""
        words = text.split()
        vector = np.zeros(self.vector_dim)

        if not words:
            return vector

        # Term Frequency
        tf = defaultdict(int)
        for word in words:
            tf[word] += 1

        # Simple IDF approximation: log(total_docs / doc_freq)
        # Since we don't have document frequency, use word rarity in current text
        total_words = len(words)
        idf = {}
        for word in set(words):
            idf[word] = np.log(total_words / (tf[word] + 1)) + 1  # +1 to avoid division by zero

        # Create vector
        for word in words:
            self.vocab.add(word)
            self.word_counts[word] += 1
            self.vocab_size = len(self.vocab)

            word_hash = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.vector_dim
            tfidf = (tf[word] / total_words) * idf[word]
            vector[word_hash] = max(vector[word_hash], tfidf)  # Keep max TF-IDF for each position

        return vector

    def get_vectorizer_stats(self) -> dict:
        """Get statistics about the vectorizer"""
        return {
            'vocab_size': self.vocab_size,
            'vector_dim': self.vector_dim,
            'unique_words': len(self.vocab),
            'total_word_occurrences': sum(self.word_counts.values()),
            'most_common_words': sorted(self.word_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        }

    def reset_stats(self) -> None:
        """Reset vectorizer statistics"""
        self.vocab = set()
        self.vocab_size = 0
        self.word_counts = defaultdict(int)

class SegmentVectorizer:
    """Specialized vectorizer for narrative segments with additional features"""

    def __init__(self, base_vectorizer: Optional[TextVectorizer] = None):
        self.base_vectorizer = base_vectorizer or TextVectorizer()
        self.segment_stats = defaultdict(int)

    def segment_to_vector(self, segment_text: str, segment_features: Dict[str, float]) -> np.ndarray:
        """
        Convert narrative segment to vector including textual and feature information

        Args:
            segment_text: Text content of the segment
            segment_features: Dictionary of segment features

        Returns:
            Combined vector representation
        """
        # Get base text vector
        text_vector = self.base_vectorizer.text_to_vector(segment_text)

        # Create feature vector (normalize features)
        feature_names = ['length', 'unique_words', 'sentence_count', 'character_names']
        feature_vector = np.zeros(len(feature_names))

        for i, feature_name in enumerate(feature_names):
            if feature_name in segment_features:
                # Normalize feature values
                normalized_value = segment_features[feature_name] / 100.0  # Simple normalization
                feature_vector[i] = min(normalized_value, 1.0)

        # Combine text and feature vectors
        combined_vector = np.concatenate([
            text_vector,
            feature_vector
        ])

        # Ensure consistent dimension
        if len(combined_vector) > self.base_vectorizer.vector_dim:
            combined_vector = combined_vector[:self.base_vectorizer.vector_dim]
        elif len(combined_vector) < self.base_vectorizer.vector_dim:
            padding = np.zeros(self.base_vectorizer.vector_dim - len(combined_vector))
            combined_vector = np.concatenate([combined_vector, padding])

        return combined_vector

    def update_segment_stats(self, segment_features: Dict[str, float]) -> None:
        """Update statistics based on segment features"""
        for feature, value in segment_features.items():
            self.segment_stats[feature] += value
