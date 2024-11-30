# examples/basic_search.py
"""
Basic example of using the Grant Finder package to search for opportunities.
"""

from pathlib import Path
from grant_finder.main import run_grant_search
from grant_finder.types import GrantSearchError
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_basic_search():
    try:
        # Define search parameters
        params = {
            "company_focus": "Artificial Intelligence and Machine Learning",
            "organization_focus": "DoD",
            "company_context_path": "path/to/company/docs",
            "funding_sources_path": "path/to/sources.csv",
            "output_dir": Path("path/to/output")
        }
        
        # Run search
        results = run_grant_search(**params)
        
        # Process results
        print("\nSearch Results Summary:")
        print(f"Total Opportunities Found: {len(results['grant_opportunities'])}")
        print("\nTop Opportunities:")
        
        # Sort opportunities by alignment score
        sorted_opps = sorted(
            results['grant_opportunities'],
            key=lambda x: x['alignment_score'],
            reverse=True
        )
        
        for i, opp in enumerate(sorted_opps[:5], 1):
            print(f"\n{i}. {opp['title']}")
            print(f"   Topic ID: {opp['topic_id']}")
            print(f"   Alignment Score: {opp['alignment_score']}")
            print(f"   Deadline: {opp['deadline']}")
            
    except GrantSearchError as e:
        logger.error(f"Search failed: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    run_basic_search()