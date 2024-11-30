# tests/test_tools.py
import pytest
from grant_finder.tools import CompanyDocumentTool, GrantSearchTool
from grant_finder.types import DocumentSearchResult, WebSearchResult

def test_company_document_tool(temp_dir):
    tool = CompanyDocumentTool(
        directory_path=str(temp_dir / "company_docs"),
        logger=Mock()
    )
    
    results = tool._run("AI capabilities")
    assert isinstance(results, list)
    assert all(isinstance(r, DocumentSearchResult) for r in results)

def test_grant_search_tool(mock_llm):
    tool = GrantSearchTool(api_key="test_key", logger=Mock())
    
    results = tool._run("SBIR AI defense")
    assert isinstance(results, list)
    assert all(isinstance(r, WebSearchResult) for r in results)