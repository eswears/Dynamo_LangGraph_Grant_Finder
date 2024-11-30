# examples/custom_tools.py
"""
Example of extending Grant Finder with custom tools.
"""

from typing import List, Dict, Any
from langchain.tools import BaseTool
from grant_finder.types import WebSearchResult, GrantSearchError
from grant_finder.main import run_grant_search
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomGrantDatabaseTool(BaseTool):
    """Example custom tool for searching a proprietary grant database"""
    
    name = "custom_grant_database"
    description = "Searches proprietary grant database for opportunities"
    
    def __init__(self, api_key: str):
        super().__init__()
        self.api_key = api_key
    
    def _run(self, query: str) -> List[WebSearchResult]:
        try:
            # Simulate database search
            # In practice, replace with actual API calls
            mock_results = [
                WebSearchResult(
                    title="Custom Grant Opportunity",
                    url="https://example.com/grant/123",
                    snippet="AI research grant opportunity",
                    score=0.95
                )
            ]
            return mock_results
            
        except Exception as e:
            raise GrantSearchError(f"Custom database search failed: {str(e)}")

def run_custom_search():
    try:
        # Create custom tool
        custom_tool = CustomGrantDatabaseTool(api_key="your_api_key")
        
        # Add custom tool to search parameters
        params = {
            "company_focus": "Quantum Computing",
            "organization_focus": "DOE",
            "company_context_path": "path/to/company/docs",
            "funding_sources_path": "path/to/sources.csv",
            "output_dir": "path/to/output",
            "additional_tools": [custom_tool]
        }
        
        # Run search with custom tool
        results = run_grant_search(**params)
        
        # Process results
        print("\nCustom Search Results:")
        print(f"Total Opportunities: {len(results['grant_opportunities'])}")
        
        # Additional custom processing...
        
    except GrantSearchError as e:
        logger.error(f"Custom search failed: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    run_custom_search()