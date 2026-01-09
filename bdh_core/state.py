"""
BDH Core State Management Module

Implements the core BDH-inspired state management with persistent internal state,
selective updates, and incremental belief formation.
"""

import numpy as np
from typing import List, Set
from dataclasses import dataclass

@dataclass
class BDHStateConfig:
    """Configuration for BDH state"""
    state_dim: int = 128
    importance_threshold: float = 0.5
    confidence_threshold: float = 0.3
    max_learning_rate: float = 0.5
    learning_rate_scale: float = 0.1

class BDHState:
    """Core BDH-inspired state management with persistent internal state"""

    def __init__(self, config: BDHStateConfig = BDHStateConfig()):
        self.config = config
        self.persistent_state = np.zeros(config.state_dim)
        self.update_counter = 0
        self.state_history: List[np.ndarray] = []
        self.last_update_positions: Set[int] = set()
        self.evidence_dim = config.state_dim  # Track expected evidence dimension

    def selective_update(self, new_info: np.ndarray, importance: float) -> None:
        """Perform selective update based on importance threshold"""
        if importance > self.config.importance_threshold:
            update_mask = np.random.random(self.config.state_dim) < importance
            self.persistent_state[update_mask] = new_info[update_mask]
            self.update_counter += 1
            self.last_update_positions.update(np.where(update_mask)[0])
            self.state_history.append(self.persistent_state.copy())

    def incremental_belief_update(self, evidence: np.ndarray, confidence: float) -> None:
        """Incrementally update beliefs based on new evidence"""
        if confidence > self.config.confidence_threshold:
            learning_rate = min(self.config.learning_rate_scale * confidence,
                               self.config.max_learning_rate)
            self.persistent_state = (1 - learning_rate) * self.persistent_state + learning_rate * evidence

    def get_current_state(self) -> np.ndarray:
        """Get current state representation"""
        return self.persistent_state.copy()

    def get_state_stats(self) -> dict:
        """Get statistics about the current state"""
        state = self.get_current_state()
        return {
            'mean': float(np.mean(state)),
            'std': float(np.std(state)),
            'variance': float(np.var(state)),
            'min': float(np.min(state)),
            'max': float(np.max(state)),
            'updates': self.update_counter,
            'update_positions': len(self.last_update_positions)
        }

    def reset_state(self) -> None:
        """Reset the state to initial conditions"""
        self.persistent_state = np.zeros(self.config.state_dim)
        self.update_counter = 0
        self.state_history = []
        self.last_update_positions = set()
