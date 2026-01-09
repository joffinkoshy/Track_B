"""
Backstory Processing Module

Handles parsing and analysis of character backstories, extracting key elements
like events, beliefs, motivations, and assumptions.
"""

import re
from typing import Dict, List, Optional
from dataclasses import dataclass
import numpy as np

@dataclass
class CharacterBackstory:
    """Represents a character backstory with structured key elements"""
    character_name: str
    early_events: List[str]
    beliefs: List[str]
    motivations: List[str]
    assumptions: List[str]
    raw_text: str
    processed_text: str
    metadata: Dict[str, str]

class BackstoryAnalyzer:
    """Analyzes character backstories and extracts key elements"""

    def __init__(self):
        # Enhanced patterns for extracting different backstory elements
        self.patterns = {
            'events': [
                r'(event|happened|occurred|experienced|when [a-z]+ was young|during childhood|growing up)',
                r'(remember|recall|look back|childhood memory|early life)'
            ],
            'beliefs': [
                r'(believed|thought|assumed|considered|was convinced|held the belief|thought that)',
                r'(belief|conviction|opinion|perspective|view|outlook)'
            ],
            'motivations': [
                r'(wanted|desired|sought|dreamed|hoped|aspired|longed|wished|yearned)',
                r'(goal|ambition|aspiration|drive|motivation|purpose|mission)'
            ],
            'assumptions': [
                r'(assumed|presumed|supposed|took for granted|believed without question)',
                r'(assumption|presumption|preconception|bias|stereotype)'
            ]
        }

        # Additional context patterns
        self.context_patterns = {
            'family': r'(family|parents|mother|father|siblings|grandparents|relatives)',
            'education': r'(school|education|learned|taught|studied|university|college)',
            'trauma': r'(trauma|tragedy|loss|hardship|difficulty|challenge|struggle)',
            'achievement': r'(achievement|success|accomplishment|victory|triumph|pride)'
        }

    def parse_backstory(self, backstory_text: str, character_name: str,
                       metadata: Optional[Dict[str, str]] = None) -> CharacterBackstory:
        """Parse backstory text into structured components with enhanced analysis"""
        processed_text = self._preprocess_text(backstory_text)

        # Extract key elements using enhanced pattern matching
        early_events = self._extract_elements_with_context(processed_text, 'events')
        beliefs = self._extract_elements_with_context(processed_text, 'beliefs')
        motivations = self._extract_elements_with_context(processed_text, 'motivations')
        assumptions = self._extract_elements_with_context(processed_text, 'assumptions')

        # Extract additional context information
        context_info = self._extract_context_info(processed_text)

        # Create metadata
        backstory_metadata = {
            'character_name': character_name,
            'text_length': str(len(backstory_text)),
            'processed_length': str(len(processed_text)),
            **context_info
        }

        if metadata:
            backstory_metadata.update(metadata)

        return CharacterBackstory(
            character_name=character_name,
            early_events=early_events,
            beliefs=beliefs,
            motivations=motivations,
            assumptions=assumptions,
            raw_text=backstory_text,
            processed_text=processed_text,
            metadata=backstory_metadata
        )

    def _preprocess_text(self, text: str) -> str:
        """Enhanced text preprocessing that preserves some structure"""
        # Keep basic punctuation for better sentence detection
        text = text.lower()
        text = re.sub(r'[^\w\s.,;:!?]', '', text)  # Keep basic punctuation
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        return text.strip()

    def _extract_elements_with_context(self, text: str, element_type: str) -> List[str]:
        """Extract elements using multiple patterns and context awareness"""
        patterns = self.patterns.get(element_type, [])
        extracted = []

        if not patterns:
            return extracted

        # Split into sentences while preserving structure
        sentences = self._split_into_sentences(text)

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Check if sentence matches any pattern for this element type
            for pattern in patterns:
                if re.search(pattern, sentence):
                    # Clean up the extracted sentence
                    clean_sentence = re.sub(r'\s+', ' ', sentence)
                    extracted.append(clean_sentence)
                    break  # Only add once per sentence

        return extracted[:5]  # Limit to top 5 for each category

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using punctuation"""
        # Use punctuation followed by whitespace or end of string as sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _extract_context_info(self, text: str) -> Dict[str, int]:
        """Extract additional context information using predefined patterns"""
        context_info = {}

        for context_type, pattern in self.context_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            context_info[f'has_{context_type}'] = str(len(matches) > 0)
            context_info[f'{context_type}_mentions'] = str(len(matches))

        return context_info

    def get_analysis_stats(self, backstory: CharacterBackstory) -> dict:
        """Get statistics about the backstory analysis"""
        return {
            'events': len(backstory.early_events),
            'beliefs': len(backstory.beliefs),
            'motivations': len(backstory.motivations),
            'assumptions': len(backstory.assumptions),
            'total_elements': (len(backstory.early_events) + len(backstory.beliefs) +
                              len(backstory.motivations) + len(backstory.assumptions)),
            'context_info': len(backstory.metadata) - 3  # Exclude basic metadata
        }


# hello
