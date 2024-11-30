# types.py
from typing import TypedDict, Annotated, Sequence, Dict, Any, List, Optional
from typing_extensions import TypeVar
from langchain_core.messages import BaseMessage
from datetime import datetime
from pydantic import BaseModel, Field
from pathlib import Path

# Existing Pydantic models remain the same
class AgentConfig(BaseModel):
    name: str = Field(..., description="Name of the agent")
    role: str = Field(..., description="Role/title of the agent")
    goal: str = Field(..., description="Primary goals and objectives of the agent")
    backstory: str = Field(..., description="Background and expertise of the agent")

# New state management types for LangGraph
class CompanyProfileState(BaseModel):
    vision: str = Field(..., description="Company vision and mission")
    technical_focus: str = Field(..., description="Company's technical focus areas")
    website: str = Field(..., description="Company website URL")
    sbir_experience: str = Field(..., description="SBIR/STTR Program/Project Experience")
    innovations: str = Field(..., description="Key innovations and capabilities")
    technical_experience: str = Field(..., description="Technical experience and past performance")

class SearchRequirementsState(BaseModel):
    technical_requirements: List[str] = Field(default_factory=list)
    innovation_areas: List[str] = Field(default_factory=list)
    competitive_advantages: List[str] = Field(default_factory=list)
    target_phases: List[str] = Field(default_factory=list)

class GrantOpportunityState(BaseModel):
    topic_id: str
    title: str
    description: str
    technical_requirements: List[str] = Field(default_factory=list)
    alignment_score: float
    award_amount: str
    deadline: str
    focus_areas: List[str] = Field(default_factory=list)
    url: str
    contact_info: str = ""
    justification: str = ""

class FundingSourceState(BaseModel):
    name: str
    url: str
    was_searched: bool = False
    search_successful: bool = False
    error_message: Optional[str] = None
    grants_found: List[GrantOpportunityState] = Field(default_factory=list)
    search_timestamp: Optional[str] = None

# Main graph state
class GrantFinderState(BaseModel):
    messages: Sequence[BaseMessage] = Field(default_factory=list)
    company_profile: CompanyProfileState = Field(default_factory=CompanyProfileState)
    search_requirements: SearchRequirementsState = Field(default_factory=SearchRequirementsState)
    funding_sources: Dict[str, FundingSourceState] = Field(default_factory=dict)
    grant_opportunities: List[GrantOpportunityState] = Field(default_factory=list)
    final_report: Dict[str, Any] = Field(default_factory=dict)
    config: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    timestamp: str = Field(..., description="Timestamp of state creation")

    class Config:
        arbitrary_types_allowed = True

# Configuration types
class LogConfig(TypedDict):
    level: str
    file: str
    system_logger: Any
    agent_logger: Any

class OutputConfig(TypedDict):
    format: str
    save_to_file: bool
    output_directory: Path

class UserInputConfig(TypedDict):
    company_focus: str
    organization_focus: str
    company_context_path: str
    funding_sources_path: str

# Tool response types
class DocumentSearchResult(TypedDict):
    content: str
    metadata: Dict[str, Any]
    relevance_score: float

class WebSearchResult(TypedDict):
    title: str
    url: str
    snippet: str
    score: float

# Error types
class GrantSearchError(Exception):
    """Custom exception for grant searching errors"""
    pass

class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass