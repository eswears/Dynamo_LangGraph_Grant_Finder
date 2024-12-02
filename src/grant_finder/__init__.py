"""
Grant Finder - A LangGraph-based system for finding and analyzing grant opportunities
"""

from grant_finder.main import main
from grant_finder.types import (
    GrantFinderState,
    GrantOpportunityState,
    GrantSearchError
)

__version__ = "0.1.0"

__all__ = [
    "main",
    "GrantFinderState",
    "GrantOpportunityState",
    "GrantSearchError"
]