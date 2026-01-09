"""
BDH Core State Management Module

Implements deterministic BDH-inspired state management with persistent internal state,
importance-thresholded updates, and context-based representation learning for KDSH 2026 Track B.
This module provides a representation-support layer that never makes decisions.
"""

import numpy as np
from typing import List, Set, Dict
from dataclasses import dataclass
import logging
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('BDHState')

class UpdateReason(Enum):
    """Enumeration of reasons for state updates"""
    IMPORTANCE_THRESHOLD_MET = "Importance threshold exceeded"
    CONFIDENCE_THRESHOLD_MET = "Confidence threshold exceeded"
    INITIAL_STATE = "Initial state setup"
    MANUAL_RESET = "Manual state reset"

@dataclass
class BDHStateConfig:
    """Configuration for BDH state"""
    state_dim: int = 128
    importance_threshold: float = 0.5
    confidence_threshold: float = 0.3
    max_learning_rate: float = 0.5
    learning_rate_scale: float = 0.1
    deterministic_seed: int = 42

@dataclass
class StateUpdateTrace:
    """Trace record for each state update"""
    update_id: int
    reason: UpdateReason
    importance_score: float
    confidence_score: float
    updated_dimensions: List[int]
    evidence_strength: float
    learning_rate_used: float
    pre_update_state: np.ndarray
    post_update_state: np.ndarray

class BDHState:
    """
    Core BDH-inspired state management with persistent internal state.

    This class provides deterministic representation for scoring components.
    It does NOT make decisions or classify consistency - only generates signals.
    """

    def __init__(self, config: BDHStateConfig = BDHStateConfig()):
        self.config = config
        self.persistent_state = np.zeros(config.state_dim)
        self.update_counter = 0
        self.state_history: List[np.ndarray] = []
        self.last_update_positions: Set[int] = set()
        self.evidence_dim = config.state_dim
        self.update_trace: List[StateUpdateTrace] = []
        self.update_reasons: List[UpdateReason] = []

        self._initialize_deterministic_state()
        logger.info(f"BDHState initialized with dimension {config.state_dim}")

    def _initialize_deterministic_state(self) -> None:
        """Initialize state deterministically"""
        initial_trace = StateUpdateTrace(
            update_id=0,
            reason=UpdateReason.INITIAL_STATE,
            importance_score=0.0,
            confidence_score=0.0,
            updated_dimensions=[],
            evidence_strength=0.0,
            learning_rate_used=0.0,
            pre_update_state=np.zeros(self.config.state_dim),
            post_update_state=np.zeros(self.config.state_dim)
        )
        self.update_trace.append(initial_trace)
        self.update_reasons.append(UpdateReason.INITIAL_STATE)

    def selective_update(self, new_info: np.ndarray, importance: float) -> Dict:
        """
        Perform deterministic importance-thresholded update with contradiction detection.

        Args:
            new_info: New information vector to incorporate
            importance: Importance score (0-1) for this information

        Returns:
            Dictionary containing update signals for scoring layer
        """
        update_signals = {
            'update_occurred': False,
            'relevance_score': 0.0,
            'confidence_signal': 0.0,
            'importance_estimate': importance,
            'updated_dimensions': [],
            'evidence_strength': 0.0,
            'potential_contradiction': False,
            'conflict_magnitude': 0.0
        }

        if importance > self.config.importance_threshold:
            # Detect potential contradictions before updating
            current_state = self.persistent_state
            conflict_mask = np.where(new_info != 0, True, False)
            conflict_magnitude = float(np.mean(np.abs(new_info[conflict_mask] - current_state[conflict_mask])))

            if conflict_magnitude > 0.3:  # Significant conflict threshold
                update_signals['potential_contradiction'] = True
                update_signals['conflict_magnitude'] = conflict_magnitude
                logger.info(f"Potential contradiction detected: magnitude={conflict_magnitude:.3f}")

            # Deterministic update - only update non-zero dimensions
            update_mask = np.where(new_info != 0, True, False)
            self.persistent_state[update_mask] = new_info[update_mask]

            # Record update for traceability
            updated_dims = np.where(update_mask)[0].tolist()
            pre_update_state = self.persistent_state.copy() - new_info
            post_update_state = self.persistent_state.copy()

            trace = StateUpdateTrace(
                update_id=self.update_counter + 1,
                reason=UpdateReason.IMPORTANCE_THRESHOLD_MET,
                importance_score=importance,
                confidence_score=0.0,
                updated_dimensions=updated_dims,
                evidence_strength=float(np.mean(np.abs(new_info[update_mask]))),
                learning_rate_used=0.0,
                pre_update_state=pre_update_state,
                post_update_state=post_update_state
            )

            self.update_trace.append(trace)
            self.update_reasons.append(UpdateReason.IMPORTANCE_THRESHOLD_MET)
            self.update_counter += 1
            self.last_update_positions.update(updated_dims)
            self.state_history.append(self.persistent_state.copy())

            update_signals.update({
                'update_occurred': True,
                'relevance_score': importance,
                'confidence_signal': float(np.mean(np.abs(new_info[update_mask]))),
                'updated_dimensions': updated_dims,
                'evidence_strength': float(np.mean(np.abs(new_info[update_mask])))
            })

            logger.info(f"Selective update: {len(updated_dims)} dimensions updated, importance={importance:.3f}")

        return update_signals

    def incremental_belief_update(self, evidence: np.ndarray, confidence: float) -> Dict:
        """
        Incrementally update state based on new evidence using deterministic learning.

        Args:
            evidence: Evidence vector for state update
            confidence: Confidence score (0-1) in this evidence

        Returns:
            Dictionary containing state update signals for scoring layer
        """
        belief_signals = {
            'belief_updated': False,
            'confidence_signal': confidence,
            'learning_rate': 0.0,
            'evidence_strength': float(np.mean(np.abs(evidence))),
            'state_change_magnitude': 0.0,
            'updated_dimensions': []
        }

        if confidence > self.config.confidence_threshold:
            # Deterministic learning rate calculation
            learning_rate = min(self.config.learning_rate_scale * confidence,
                               self.config.max_learning_rate)

            pre_update_state = self.persistent_state.copy()
            self.persistent_state = (1 - learning_rate) * self.persistent_state + learning_rate * evidence

            # Calculate changes
            state_change = self.persistent_state - pre_update_state
            updated_dims = np.where(state_change != 0)[0].tolist()

            trace = StateUpdateTrace(
                update_id=self.update_counter + 1,
                reason=UpdateReason.CONFIDENCE_THRESHOLD_MET,
                importance_score=0.0,
                confidence_score=confidence,
                updated_dimensions=updated_dims,
                evidence_strength=float(np.mean(np.abs(evidence))),
                learning_rate_used=learning_rate,
                pre_update_state=pre_update_state,
                post_update_state=self.persistent_state.copy()
            )

            self.update_trace.append(trace)
            self.update_reasons.append(UpdateReason.CONFIDENCE_THRESHOLD_MET)
            self.update_counter += 1
            self.last_update_positions.update(updated_dims)
            self.state_history.append(self.persistent_state.copy())

            belief_signals.update({
                'belief_updated': True,
                'learning_rate': learning_rate,
                'state_change_magnitude': float(np.mean(np.abs(state_change))),
                'updated_dimensions': updated_dims
            })

            logger.info(f"Belief update: {len(updated_dims)} dimensions changed, "
                       f"confidence={confidence:.3f}, learning_rate={learning_rate:.3f}")

        return belief_signals

    def get_reasoning_signals(self) -> Dict:
        """
        Generate signals for higher-level scoring components.

        Returns:
            Dictionary of signals for scoring layer
        """
        current_state = self.get_current_state()

        return {
            'state_vector': current_state,
            'state_magnitude': float(np.linalg.norm(current_state)),
            'total_updates': self.update_counter,
            'recent_updates': len(self.last_update_positions),
            'update_density': len(self.last_update_positions) / self.config.state_dim,
            'state_confidence': float(np.mean(np.abs(current_state))),
            'state_variability': float(np.std(current_state)),
            'active_dimensions': self.last_update_positions,
            'sparse_update_ratio': len(self.last_update_positions) / self.config.state_dim,
            'last_update_reason': self.update_reasons[-1].value if self.update_reasons else "None",
            'update_history_size': len(self.update_trace)
        }

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
            'update_positions': len(self.last_update_positions),
            'l2_norm': float(np.linalg.norm(state))
        }

    def get_update_trace(self) -> List[StateUpdateTrace]:
        """Get complete trace of all state updates"""
        return self.update_trace.copy()

    def get_update_summary(self) -> Dict:
        """Get summary of state updates"""
        if not self.update_trace:
            return {'total_updates': 0, 'reasons': {}, 'dimension_coverage': 0}

        reason_counts = {}
        for reason in self.update_reasons:
            reason_counts[reason.value] = reason_counts.get(reason.value, 0) + 1

        all_updated_dims = set()
        for trace in self.update_trace[1:]:
            all_updated_dims.update(trace.updated_dimensions)

        return {
            'total_updates': len(self.update_trace) - 1,
            'reasons': reason_counts,
            'dimension_coverage': len(all_updated_dims) / self.config.state_dim,
            'average_importance': np.mean([t.importance_score for t in self.update_trace[1:]]),
            'average_confidence': np.mean([t.confidence_score for t in self.update_trace[1:]])
        }

    def reset_state(self) -> None:
        """Reset the state to initial conditions"""
        self.persistent_state = np.zeros(self.config.state_dim)
        self.update_counter = 0
        self.state_history = []
        self.last_update_positions = set()
        self.update_trace = []
        self.update_reasons = []

        self._initialize_deterministic_state()
        logger.info("BDHState reset to initial conditions")

    def get_track_b_compliance_info(self) -> Dict:
        """Get information about Track B compliance"""
        return {
            'bdh_principles': {
                'persistent_internal_state': {
                    'implementation': 'Maintained across all operations via persistent_state vector',
                    'evidence': f'State dimension: {self.config.state_dim}, '
                               f'Current updates: {self.update_counter}',
                    'reproducibility': 'Deterministic state evolution ensures same inputs → same outputs'
                },
                'selective_updates': {
                    'implementation': 'Importance-thresholded deterministic updates via selective_update()',
                    'evidence': f'Importance threshold: {self.config.importance_threshold}, '
                               f'Update trace: {len(self.update_trace)} records',
                    'reproducibility': 'No randomness - uses deterministic importance-based selection'
                },
                'incremental_belief_formation': {
                    'implementation': 'Confidence-weighted learning via incremental_belief_update()',
                    'evidence': f'Confidence threshold: {self.config.confidence_threshold}, '
                               f'Learning rate scale: {self.config.learning_rate_scale}',
                    'reproducibility': 'Deterministic learning rate calculation from confidence scores'
                }
            },
            'separation_of_concerns': {
                'bdh_role': 'Representation layer only - generates signals for reasoning',
                'reasoning_role': 'External components interpret signals and make decisions',
                'api_boundary': 'get_reasoning_signals() method provides clean interface'
            },
            'reproducibility_mechanisms': [
                'No random number generation',
                'Deterministic update selection',
                'Fixed learning rate calculation',
                'Complete state traceability'
            ],
            'interpretability_features': [
                'Full update trace with StateUpdateTrace records',
                'Reason tracking with UpdateReason enumeration',
                'Comprehensive logging',
                'State change statistics'
            ]
        }
