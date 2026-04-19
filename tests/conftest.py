"""
tests/conftest.py
-----------------
Shared pytest fixtures for cascade-mind test suite.
"""
import os
import pytest

# Disable LLM for all tests unless explicitly overridden
os.environ.setdefault("LLM_SIMULATOR_ENABLED", "false")


@pytest.fixture
def env_url() -> str:
    return os.environ.get("CASCADE_MIND_URL", "http://localhost:8000")


@pytest.fixture
def hf_token() -> str:
    return os.environ.get("HF_TOKEN", "")
