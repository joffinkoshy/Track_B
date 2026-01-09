"""
BDH Core Module Tests

Tests for the core BDH state management functionality.

Updated to test the refactored deterministic BDH state with separation of concerns.
"""

import pytest
import numpy as np
from bdh_core.state import BDHState, BDHStateConfig, UpdateReason

class TestBDHState:
    """Test BDHState class functionality"""

    def test_initialization(self):
        """Test BDHState initialization"""
        config = BDHStateConfig(state_dim=128)
        state = BDHState(config)

        assert state.persistent_state.shape == (128,)
        assert np.all(state.persistent_state == 0)  # Initial state should be zeros
        assert state.update_counter == 0
        assert len(state.state_history) == 0
        assert len(state.last_update_positions) == 0
        assert len(state.update_trace) == 1  # Initial trace record
        assert state.update_reasons == [UpdateReason.INITIAL_STATE]

    def test_selective_update_deterministic(self):
        """Test deterministic selective update functionality"""
        config = BDHStateConfig(state_dim=64, importance_threshold=0.5)
        state = BDHState(config)

        # Create test input with some non-zero dimensions
        new_info = np.zeros(64)
        new_info[10:20] = 0.8  # Only update dimensions 10-19

        # Test with high importance (should update deterministically)
        result = state.selective_update(new_info, importance=0.7)

        # Verify signals returned
        assert result['update_occurred'] == True
        assert result['relevance_score'] == 0.7
        assert result['importance_estimate'] == 0.7
        assert len(result['updated_dimensions']) == 10  # Dimensions 10-19
        assert result['evidence_strength'] > 0

        # Verify state updates
        assert state.update_counter == 1
        assert len(state.state_history) == 1
        assert len(state.last_update_positions) == 10
        assert state.update_reasons[-1] == UpdateReason.IMPORTANCE_THRESHOLD_MET

        # Test with low importance (should not update)
        initial_counter = state.update_counter
        result = state.selective_update(new_info, importance=0.3)
        assert result['update_occurred'] == False
        assert state.update_counter == initial_counter  # No change

    def test_incremental_belief_update_deterministic(self):
        """Test deterministic incremental belief update functionality"""
        config = BDHStateConfig(state_dim=32, confidence_threshold=0.3)
        state = BDHState(config)

        # Create test evidence
        evidence = np.ones(32) * 0.6

        # Test with high confidence (should update deterministically)
        result = state.incremental_belief_update(evidence, confidence=0.5)

        # Verify signals returned
        assert result['belief_updated'] == True
        assert result['confidence_signal'] == 0.5
        assert result['learning_rate'] > 0
        assert result['evidence_strength'] > 0
        assert result['state_change_magnitude'] > 0
        assert len(result['updated_dimensions']) == 32  # All dimensions updated

        # Verify state changes
        assert not np.all(state.persistent_state == 0)  # State should change
        assert state.update_reasons[-1] == UpdateReason.CONFIDENCE_THRESHOLD_MET

        # Test with low confidence (should not update)
        initial_state = state.persistent_state.copy()
        result = state.incremental_belief_update(evidence, confidence=0.2)
        assert result['belief_updated'] == False
        assert np.allclose(state.persistent_state, initial_state)  # No change

    def test_reasoning_signals(self):
        """Test reasoning signals generation"""
        config = BDHStateConfig(state_dim=64)
        state = BDHState(config)

        # Add some updates
        test_vector = np.zeros(64)
        test_vector[5:15] = 0.8
        state.selective_update(test_vector, importance=0.8)

        # Get reasoning signals
        signals = state.get_reasoning_signals()

        # Verify signal structure
        assert 'state_vector' in signals
        assert 'state_magnitude' in signals
        assert 'total_updates' in signals
        assert 'recent_updates' in signals
        assert 'state_confidence' in signals
        assert 'active_dimensions' in signals
        assert 'last_update_reason' in signals

        # Verify signal values
        assert signals['total_updates'] == 1
        assert signals['recent_updates'] == 10  # Dimensions 5-14
        assert signals['last_update_reason'] == "Importance threshold exceeded"

    def test_state_stats(self):
        """Test state statistics calculation"""
        config = BDHStateConfig(state_dim=64)
        state = BDHState(config)

        # Add some updates
        test_vector = np.zeros(64)
        test_vector[10:20] = 0.5
        state.selective_update(test_vector, importance=0.8)

        stats = state.get_state_stats()
        assert 'mean' in stats
        assert 'std' in stats
        assert 'variance' in stats
        assert 'updates' in stats
        assert 'l2_norm' in stats
        assert stats['updates'] == 1

    def test_update_trace(self):
        """Test update trace functionality"""
        config = BDHStateConfig(state_dim=32)
        state = BDHState(config)

        # Add updates
        test_vector = np.zeros(32)
        test_vector[5:10] = 0.6
        state.selective_update(test_vector, importance=0.7)

        # Get update trace
        trace = state.get_update_trace()
        assert len(trace) == 2  # Initial + one update
        assert trace[1].update_id == 1
        assert trace[1].reason == UpdateReason.IMPORTANCE_THRESHOLD_MET
        assert len(trace[1].updated_dimensions) == 5
        assert trace[1].importance_score == 0.7

    def test_update_summary(self):
        """Test update summary functionality"""
        config = BDHStateConfig(state_dim=32)
        state = BDHState(config)

        # Add different types of updates
        test_vector = np.zeros(32)
        test_vector[5:10] = 0.6
        state.selective_update(test_vector, importance=0.7)
        state.incremental_belief_update(test_vector, confidence=0.6)

        summary = state.get_update_summary()
        assert summary['total_updates'] == 2
        assert 'Importance threshold exceeded' in summary['reasons']
        assert 'Confidence threshold exceeded' in summary['reasons']
        assert summary['dimension_coverage'] > 0

    def test_reset_state(self):
        """Test state reset functionality"""
        config = BDHStateConfig(state_dim=32)
        state = BDHState(config)

        # Add updates
        test_vector = np.ones(32) * 0.5
        state.selective_update(test_vector, importance=0.8)
        state.incremental_belief_update(test_vector, confidence=0.6)

        # Reset and verify
        state.reset_state()
        assert np.all(state.persistent_state == 0)
        assert state.update_counter == 0
        assert len(state.state_history) == 0
        assert len(state.last_update_positions) == 0
        assert len(state.update_trace) == 1  # Only initial trace
        assert state.update_reasons == [UpdateReason.INITIAL_STATE]

    def test_track_b_compliance_info(self):
        """Test Track B compliance information"""
        config = BDHStateConfig(state_dim=64)
        state = BDHState(config)

        compliance = state.get_track_b_compliance_info()

        # Verify structure
        assert 'bdh_principles' in compliance
        assert 'separation_of_concerns' in compliance
        assert 'reproducibility_mechanisms' in compliance
        assert 'interpretability_features' in compliance

        # Verify BDH principles
        principles = compliance['bdh_principles']
        assert 'persistent_internal_state' in principles
        assert 'selective_updates' in principles
        assert 'incremental_belief_formation' in principles

        # Verify separation of concerns
        separation = compliance['separation_of_concerns']
        assert separation['bdh_role'] == 'Representation layer only - generates signals for reasoning'
        assert separation['api_boundary'] == 'get_reasoning_signals() method provides clean interface'

    def test_deterministic_reproducibility(self):
        """Test that same inputs produce same outputs (reproducibility)"""
        config = BDHStateConfig(state_dim=32, importance_threshold=0.5)

        # Create two identical states
        state1 = BDHState(config)
        state2 = BDHState(config)

        # Apply same operations
        test_vector = np.zeros(32)
        test_vector[5:15] = 0.7

        result1 = state1.selective_update(test_vector, importance=0.8)
        result2 = state2.selective_update(test_vector, importance=0.8)

        # Verify same results
        assert result1['update_occurred'] == result2['update_occurred']
        assert np.allclose(result1['relevance_score'], result2['relevance_score'])
        assert result1['updated_dimensions'] == result2['updated_dimensions']
        assert np.allclose(state1.persistent_state, state2.persistent_state)

class TestBDHStateConfig:
    """Test BDHStateConfig class"""

    def test_default_config(self):
        """Test default configuration values"""
        config = BDHStateConfig()

        assert config.state_dim == 128
        assert config.importance_threshold == 0.5
        assert config.confidence_threshold == 0.3
        assert config.max_learning_rate == 0.5
        assert config.learning_rate_scale == 0.1
        assert config.deterministic_seed == 42  # New field

    def test_custom_config(self):
        """Test custom configuration"""
        config = BDHStateConfig(
            state_dim=256,
            importance_threshold=0.7,
            confidence_threshold=0.4,
            max_learning_rate=0.6,
            learning_rate_scale=0.15,
            deterministic_seed=123
        )

        assert config.state_dim == 256
        assert config.importance_threshold == 0.7
        assert config.confidence_threshold == 0.4
        assert config.max_learning_rate == 0.6
        assert config.learning_rate_scale == 0.15
        assert config.deterministic_seed == 123
