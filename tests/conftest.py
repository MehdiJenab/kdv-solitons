"""Shared pytest fixtures for kdv-solver tests."""

import pytest


@pytest.fixture
def default_grid_params():
    """Default grid parameters for KdV simulations."""
    return {
        "N": 256,  # number of grid points
        "L": 100.0,  # domain length
    }


@pytest.fixture
def soliton_params():
    """Default soliton parameters."""
    return {
        "kappa": 0.5,  # soliton parameter
    }


@pytest.fixture
def time_params():
    """Default time integration parameters."""
    return {
        "dt": 0.001,  # time step (will be rescaled for grid in tests)
        "t_final": 0.5,  # final time (shorter to avoid instability)
    }
