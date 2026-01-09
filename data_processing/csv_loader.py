"""
CSV Data Loader Module

Handles loading and processing of CSV data for Track B BDH implementation.
This module maintains BDH principles while working with structured backstory data.
"""

import csv
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class BackstoryElement:
    """Represents a single backstory element from CSV with BDH context"""
    id: str
    book_name: str
    character: str
    caption: str
    content: str
    label: Optional[str] = None  # Only for training data

class CSVDataLoader:
    """Loads and processes CSV data while maintaining BDH context and memory"""

    def __init__(self):
        # BDH-style context memory
        self.book_contexts: Dict[str, Dict] = {}  # Store context for each book
        self.character_contexts: Dict[str, Dict] = {}  # Store context for each character
        self.temporal_patterns: Dict[str, List[str]] = {}  # Track temporal information

        # BDH state tracking
        self.processing_history: List[Dict] = []
        self.context_evolution: Dict[str, List[Dict]] = {}

    def load_csv_data(self, file_path: str, is_training: bool = False) -> List[BackstoryElement]:
        """
        Load CSV data and create backstory elements with BDH context tracking

        Args:
            file_path: Path to CSV file
            is_training: Whether this is training data (has labels)

        Returns:
            List of BackstoryElement objects
        """
        elements = []
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                element = BackstoryElement(
                    id=row['id'],
                    book_name=row['book_name'],
                    character=row['char'],
                    caption=row['caption'] if row['caption'] else "",
                    content=row['content'],
                    label=row['label'] if is_training and 'label' in row else None
                )
                elements.append(element)
                self._update_bdh_contexts(element)

        return elements

    def _update_bdh_contexts(self, element: BackstoryElement) -> None:
        """
        Update BDH-style contexts with new element information.
        This implements incremental context building inspired by BDH principles.
        """
        # Update book context (persistent state for books)
        if element.book_name not in self.book_contexts:
            self.book_contexts[element.book_name] = {
                'characters': set(),
                'themes': set(),
                'temporal_span': [],
                'element_count': 0,
                'consistency_patterns': []
            }
        book_ctx = self.book_contexts[element.book_name]
        book_ctx['characters'].add(element.character)
        book_ctx['element_count'] += 1

        # Extract temporal information for BDH temporal reasoning
        self._extract_temporal_info(element, book_ctx)

        # Update character context (persistent state for characters)
        char_key = f"{element.book_name}_{element.character}"
        if char_key not in self.character_contexts:
            self.character_contexts[char_key] = {
                'elements': [],
                'temporal_patterns': [],
                'consistency_history': [],
                'thematic_clusters': {}
            }
        char_ctx = self.character_contexts[char_key]
        char_ctx['elements'].append(element)

        # Track context evolution (BDH state history)
        if char_key not in self.context_evolution:
            self.context_evolution[char_key] = []
        self.context_evolution[char_key].append({
            'element_id': element.id,
            'context_size': len(char_ctx['elements']),
            'temporal_info': book_ctx['temporal_span'][:]  # Copy
        })

    def _extract_temporal_info(self, element: BackstoryElement, book_ctx: Dict) -> None:
        """Extract temporal patterns for BDH-style temporal reasoning"""
        # Simple temporal indicators - would be enhanced in full implementation
        temporal_indicators = [
            '17', '18', '19', '1800', '1815', '1852', 'young', 'childhood',
            'boyhood', 'growing up', 'at twelve', 'at fifteen', 'later', 'earlier',
            'during', 'before', 'after', 'when', 'while'
        ]

        content_lower = element.content.lower()
        found_temporal = []

        for indicator in temporal_indicators:
            if indicator in content_lower:
                found_temporal.append(indicator)

        if found_temporal:
            book_ctx['temporal_span'].extend(found_temporal)
            char_key = f"{element.book_name}_{element.character}"
            if char_key in self.character_contexts:
                self.character_contexts[char_key]['temporal_patterns'].extend(found_temporal)

    def create_bdh_context(self, element: BackstoryElement) -> str:
        """
        Create contextual narrative for BDH processing.
        This builds a pseudo-narrative that provides context for the backstory element.

        Args:
            element: The backstory element to create context for

        Returns:
            Contextual narrative string for BDH processing
        """
        book_ctx = self.book_contexts.get(element.book_name, {})
        char_key = f"{element.book_name}_{element.character}"
        char_ctx = self.character_contexts.get(char_key, {})

        context_parts = []

        # Book context
        context_parts.append(f"Book: {element.book_name}.")

        # Character context
        context_parts.append(f"Character: {element.character}.")

        # Temporal context
        if 'temporal_span' in book_ctx and book_ctx['temporal_span']:
            temporal_info = list(set(book_ctx['temporal_span']))  # Remove duplicates
            if temporal_info:
                context_parts.append(f"Temporal context: {', '.join(temporal_info[:5])}.")

        # Related characters
        if 'characters' in book_ctx and len(book_ctx['characters']) > 1:
            other_chars = [c for c in book_ctx['characters'] if c != element.character]
            if other_chars:
                context_parts.append(f"Related characters: {', '.join(other_chars[:3])}.")

        # Caption as additional context
        if element.caption:
            context_parts.append(f"Context: {element.caption}.")

        # Element position in character's story
        if char_key in self.character_contexts:
            element_count = len(self.character_contexts[char_key]['elements'])
            context_parts.append(f"Element {element_count} of {element_count} in character's backstory.")

        return ' '.join(context_parts)

    def get_bdh_context_stats(self) -> Dict:
        """Get statistics about BDH context building"""
        return {
            'books_processed': len(self.book_contexts),
            'characters_processed': len(self.character_contexts),
            'total_elements': sum(len(ctx['elements']) for ctx in self.character_contexts.values()),
            'context_evolution_depth': sum(len(evol) for evol in self.context_evolution.values()),
            'temporal_patterns_found': sum(len(ctx.get('temporal_span', [])) for ctx in self.book_contexts.values())
        }

    def reset_contexts(self) -> None:
        """Reset BDH contexts (for fresh processing)"""
        self.book_contexts = {}
        self.character_contexts = {}
        self.temporal_patterns = {}
        self.processing_history = []
        self.context_evolution = {}
