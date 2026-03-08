"""Shared test fixtures."""

import pytest


@pytest.fixture
def sample_messages():
    """Sample chat messages for testing."""
    return [
        {"role": "user", "content": "Hello, how are you?"},
    ]


@pytest.fixture
def sample_rag_messages():
    """Sample RAG-oriented messages."""
    return [
        {"role": "user", "content": "What does the paper say about machine learning?"},
    ]


@pytest.fixture
def sample_latex_messages():
    """Sample LaTeX generation messages."""
    return [
        {"role": "user", "content": "Generate a LaTeX equation for the quadratic formula"},
    ]


@pytest.fixture
def sample_task_messages():
    """Sample task planning messages."""
    return [
        {"role": "user", "content": "Suggest next tasks for my research project on NLP"},
    ]


@pytest.fixture
def sample_analyze_messages():
    """Sample analysis messages."""
    return [
        {"role": "user", "content": "Analyze the key findings of this research paper"},
    ]
