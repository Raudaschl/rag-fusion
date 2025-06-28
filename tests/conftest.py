import os
import pytest
import tempfile
import shutil
from unittest.mock import Mock, patch


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Fixture to mock environment variables."""
    def _mock_env_vars(env_dict):
        for key, value in env_dict.items():
            monkeypatch.setenv(key, value)
    return _mock_env_vars


@pytest.fixture
def mock_openai_key(monkeypatch):
    """Mock OpenAI API key."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key-12345")


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response."""
    mock_response = {
        "choices": [{
            "message": {
                "content": "Query 1\nQuery 2\nQuery 3\nQuery 4"
            }
        }]
    }
    return mock_response


@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return {
        "doc1": "Climate change and economic impact.",
        "doc2": "Public health concerns due to climate change.",
        "doc3": "Climate change: A social perspective.",
        "doc4": "Technological solutions to climate change.",
        "doc5": "Policy changes needed to combat climate change."
    }


@pytest.fixture
def mock_config():
    """Mock configuration object."""
    return {
        "api_key": "test-key",
        "model": "gpt-3.5-turbo",
        "temperature": 0.7,
        "max_tokens": 150
    }


@pytest.fixture(autouse=True)
def reset_modules():
    """Reset imported modules to avoid side effects between tests."""
    yield
    # Clean up any module-level state if needed


@pytest.fixture
def capture_stdout(monkeypatch):
    """Capture stdout for testing print statements."""
    import io
    import sys
    
    captured_output = io.StringIO()
    monkeypatch.setattr(sys, 'stdout', captured_output)
    return captured_output


@pytest.fixture
def mock_file_operations(tmp_path):
    """Mock file operations with temporary directory."""
    def _create_file(filename, content):
        file_path = tmp_path / filename
        file_path.write_text(content)
        return str(file_path)
    
    return _create_file


@pytest.fixture(scope="session")
def test_data_dir():
    """Path to test data directory."""
    return os.path.join(os.path.dirname(__file__), "test_data")


@pytest.fixture
def mock_api_client():
    """Mock API client for testing."""
    client = Mock()
    client.api_key = "test-key"
    client.base_url = "https://api.example.com"
    return client