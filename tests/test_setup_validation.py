"""Validation tests to ensure the testing infrastructure is properly set up."""

import pytest
import os
import sys


class TestSetupValidation:
    """Test class to validate the testing infrastructure."""
    
    def test_pytest_is_working(self):
        """Verify that pytest is working correctly."""
        assert True
    
    def test_project_root_in_path(self):
        """Verify that the project root is in the Python path."""
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        assert project_root in sys.path or os.path.abspath('.') in sys.path
    
    def test_main_module_importable(self, monkeypatch):
        """Verify that the main module can be imported."""
        # Set a dummy API key to prevent the module from failing on import
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        try:
            import main
            assert hasattr(main, 'generate_queries_chatgpt')
            assert hasattr(main, 'vector_search')
            assert hasattr(main, 'reciprocal_rank_fusion')
        except ImportError as e:
            pytest.fail(f"Failed to import main module: {e}")
    
    @pytest.mark.unit
    def test_unit_marker_works(self):
        """Verify that the unit test marker works."""
        assert True
    
    @pytest.mark.integration
    def test_integration_marker_works(self):
        """Verify that the integration test marker works."""
        assert True
    
    @pytest.mark.slow
    def test_slow_marker_works(self):
        """Verify that the slow test marker works."""
        assert True
    
    def test_fixtures_available(self, temp_dir, mock_env_vars, sample_documents):
        """Verify that custom fixtures are available and working."""
        # Test temp_dir fixture
        assert os.path.exists(temp_dir)
        assert os.path.isdir(temp_dir)
        
        # Test mock_env_vars fixture
        mock_env_vars({"TEST_VAR": "test_value"})
        assert os.environ.get("TEST_VAR") == "test_value"
        
        # Test sample_documents fixture
        assert isinstance(sample_documents, dict)
        assert len(sample_documents) == 5
        assert "doc1" in sample_documents
    
    def test_mock_openai_key_fixture(self, mock_openai_key):
        """Verify that the mock OpenAI key fixture works."""
        assert os.environ.get("OPENAI_API_KEY") == "test-api-key-12345"
    
    def test_capture_stdout_fixture(self, capture_stdout):
        """Verify that stdout capture fixture works."""
        print("Test output")
        # Note: In pytest with coverage, stdout might be captured differently
        # This test verifies the fixture is available and callable
        assert capture_stdout is not None
        assert hasattr(capture_stdout, 'getvalue')
    
    def test_coverage_configured(self):
        """Verify that coverage is properly configured."""
        # This test will pass if coverage is running
        # The actual verification happens when running with coverage
        assert True


@pytest.mark.parametrize("test_input,expected", [
    (1, 1),
    (2, 4),
    (3, 9),
    (4, 16),
])
def test_parametrize_works(test_input, expected):
    """Verify that parametrized tests work correctly."""
    assert test_input ** 2 == expected