"""
BDH Core Module Tests

Tests for the core BDH state management functionality.
"""

import pytest
import numpy as np
from bdh_core.state import BDHState, BDHStateConfig

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

    def test_selective_update(self):
        """Test selective update functionality"""
        config = BDHStateConfig(state_dim=64, importance_threshold=0.5)
        state = BDHState(config)

        # Create test input
        new_info = np.ones(64) * 0.8

        # Test with high importance (should update)
        state.selective_update(new_info, importance=0.7)
        assert state.update_counter == 1
        assert len(state.state_history) == 1
        assert len(state.last_update_positions) > 0

        # Test with low importance (should not update)
        initial_counter = state.update_counter
        state.selective_update(new_info, importance=0.3)
        assert state.update_counter == initial_counter  # No change

    def test_incremental_belief_update(self):
        """Test incremental belief update functionality"""
        config = BDHStateConfig(state_dim=32, confidence_threshold=0.3)
        state = BDHState(config)

        # Create test evidence
        evidence = np.ones(32) * 0.6

        # Test with high confidence (should update)
        state.incremental_belief_update(evidence, confidence=0.5)
        assert not np.all(state.persistent_state == 0)  # State should change

        # Test with low confidence (should not update)
        initial_state = state.persistent_state.copy()
        state.incremental_belief_update(evidence, confidence=0.2)
        assert np.allclose(state.persistent_state, initial_state)  # No change

    def test_state_stats(self):
        """Test state statistics calculation"""
        config = BDHStateConfig(state_dim=64)
        state = BDHState(config)

        # Add some updates
        test_vector = np.random.random(64)
        state.selective_update(test_vector, importance=0.8)

        stats = state.get_state_stats()
        assert 'mean' in stats
        assert 'std' in stats
        assert 'variance' in stats
        assert 'updates' in stats
        assert stats['updates'] == 1

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

    def test_custom_config(self):
        """Test custom configuration"""
        config = BDHStateConfig(
            state_dim=256,
            importance_threshold=0.7,
            confidence_threshold=0.4,
            max_learning_rate=0.6,
            learning_rate_scale=0.15
        )

        assert config.state_dim == 256
        assert config.importance_threshold == 0.7
        assert config.confidence_threshold == 0.4
        assert config.max_learning_rate == 0.6
        assert config.learning_rate_scale == 0.15
