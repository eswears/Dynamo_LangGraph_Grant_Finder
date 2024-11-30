# tests/test_nodes.py
import pytest
from grant_finder.nodes import ProfileAnalysisNode, StrategyDevelopmentNode
from grant_finder.types import GrantFinderState

def test_profile_analysis_node(mock_llm, temp_dir, sample_config):
    node = ProfileAnalysisNode(
        company_context_path=sample_config["company_context"]["directory"],
        logger=Mock(),
        llm=mock_llm
    )
    
    initial_state = GrantFinderState(
        messages=[],
        company_profile={},
        search_requirements={},
        funding_sources={},
        grant_opportunities=[],
        final_report={},
        config=sample_config,
        errors=[],
        timestamp="20241130_120000"
    )
    
    result_state = node(initial_state)
    assert "company_profile" in result_state
    assert result_state["errors"] == []

def test_strategy_development_node(mock_llm):
    node = StrategyDevelopmentNode(logger=Mock(), llm=mock_llm)
    
    initial_state = GrantFinderState(
        messages=[],
        company_profile={"technical_focus": "AI/ML"},
        search_requirements={},
        funding_sources={},
        grant_opportunities=[],
        final_report={},
        config={"company_focus": "AI"},
        errors=[],
        timestamp="20241130_120000"
    )
    
    result_state = node(initial_state)
    assert "search_requirements" in result_state
    assert result_state["errors"] == []