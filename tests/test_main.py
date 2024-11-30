'''
Run test_main.py First
verify implementation and accuracy
Catches basic errors and bugs
Tests error handling
Runs quickly with mock data
No API costs or real network calls
More comprehensive coverage of edge cases

Run main.py second,
Tests actual real-world functionality
Uses real API calls and data
Processes real company documents
Produces actual grant recommendations
Shows real user interaction
Verifies actual output quality
'''

import pytest
from unittest.mock import Mock, patch
from pathlib import Path
import yaml
import json
from datetime import datetime

from grant_finder.main import main, load_config, get_user_input, save_results
from grant_finder.types import GrantSearchError, GrantFinderState

class TestMain:
    """Test cases for main execution flow"""
    
    @pytest.fixture
    def mock_env(self, monkeypatch, temp_dir):
        """Setup mock environment"""
        monkeypatch.setenv("OPENAI_API_KEY", "test_key")
        monkeypatch.setenv("SERPER_API_KEY", "test_key")
        
        # Create test config
        config_path = temp_dir / "config"
        config_path.mkdir()
        config = {
            "company_context": {"directory": str(temp_dir / "docs")},
            "funding_sources": {"file_path": str(temp_dir / "sources.csv")},
            "output": {"output_directory": str(temp_dir / "output")},
            "logging": {"level": "INFO", "file": "test.log"}
        }
        with open(config_path / "user_config.yaml", 'w') as f:
            yaml.dump(config, f)
            
        return config

    def test_successful_execution(self, mock_env, mock_llm, temp_dir):
        """Test successful end-to-end execution"""
        with patch('grant_finder.main.get_user_input') as mock_input:
            mock_input.side_effect = ["AI Testing", "DoD"]
            
            # Run main
            main()
            
            # Verify outputs
            output_dir = temp_dir / "output"
            assert output_dir.exists()
            
            # Check for expected output files
            files = list(output_dir.glob("*/*.json"))
            assert any("company_profile" in f.name for f in files)
            assert any("grant_opportunities" in f.name for f in files)
            assert any("final_report" in f.name for f in files)

    def test_configuration_loading(self, mock_env, temp_dir):
        """Test configuration loading"""
        config = load_config(temp_dir / "config" / "user_config.yaml")
        assert "company_context" in config
        assert "funding_sources" in config
        assert "output" in config
        
        # Test invalid config
        with pytest.raises(GrantSearchError):
            load_config(temp_dir / "nonexistent.yaml")

    def test_user_input_handling(self, capsys):
        """Test user input handling"""
        with patch('builtins.input', side_effect=["", "Custom Focus"]):
            # Test default value
            result1 = get_user_input("Test prompt", "Default")
            assert result1 == "Default"
            
            # Test custom input
            result2 = get_user_input("Test prompt", "Default")
            assert result2 == "Custom Focus"
            
        # Check prompts were displayed
        captured = capsys.readouterr()
        assert "Test prompt" in captured.out

    def test_result_saving(self, temp_dir):
        """Test result saving functionality"""
        state = GrantFinderState(
            messages=[],
            company_profile={"focus": "AI"},
            search_requirements={"requirements": ["R1"]},
            funding_sources={"source1": {"url": "http://test.com"}},
            grant_opportunities=[{
                "topic_id": "TEST-001",
                "title": "Test Grant"
            }],
            final_report={"summary": "Test"},
            config={},
            errors=[],
            timestamp="20241130_120000"
        )
        
        # Save results
        save_results(state, temp_dir, "20241130_120000")
        
        # Verify files
        assert (temp_dir / "20241130_120000" / "01_company_profile.json").exists()
        assert (temp_dir / "20241130_120000" / "summary.json").exists()
        
        # Verify content
        with open(temp_dir / "20241130_120000" / "summary.json") as f:
            summary = json.load(f)
            assert summary["total_opportunities_found"] == 1
            assert summary["errors"] == []

    def test_error_handling(self, mock_env, monkeypatch):
        """Test error handling scenarios"""
        # Test missing API key
        monkeypatch.delenv("OPENAI_API_KEY")
        with pytest.raises(GrantSearchError, match="API key not found"):
            main()
        
        # Test invalid config
        monkeypatch.setenv("OPENAI_API_KEY", "test_key")
        with patch('grant_finder.main.load_config') as mock_config:
            mock_config.side_effect = Exception("Config error")
            with pytest.raises(GrantSearchError, match="Config error"):
                main()

    @pytest.mark.asyncio
    async def test_workflow_execution(self, mock_env, mock_llm):
        """Test the LangGraph workflow execution"""
        with patch('grant_finder.main.build_graph') as mock_build:
            mock_workflow = Mock()
            mock_workflow.invoke.return_value = GrantFinderState(
                messages=[],
                company_profile={},
                search_requirements={},
                funding_sources={},
                grant_opportunities=[],
                final_report={},
                config={},
                errors=[],
                timestamp="20241130_120000"
            )
            mock_build.return_value = mock_workflow
            
            # Run main
            main()
            
            # Verify workflow was built and executed
            assert mock_build.called
            assert mock_workflow.invoke.called

    def test_logging_setup(self, mock_env, temp_dir):
        """Test logging configuration"""
        with patch('grant_finder.main.setup_logging') as mock_logging:
            main()
            assert mock_logging.called
            
            # Verify log directory
            log_dir = list(temp_dir.glob("*/logs"))[0]
            assert log_dir.exists()
            assert (log_dir / "grant_finder.log").exists()
            assert (log_dir / "agent_interactions.log").exists()