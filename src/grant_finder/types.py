# types.py
from typing import TypedDict, Annotated, Sequence, Dict, Any, List, Optional
from typing_extensions import TypeVar
from langchain_core.messages import BaseMessage
from datetime import datetime
from pathlib import Path
from langchain_core.runnables import RunnableConfig
# Update imports to use Pydantic v2
from pydantic import BaseModel, Field, ConfigDict

# Existing Pydantic models remain the same
class AgentConfig(BaseModel):
    name: str = Field(..., description="Name of the agent")
    role: str = Field(..., description="Role/title of the agent")
    goal: str = Field(..., description="Primary goals and objectives of the agent")
    backstory: str = Field(..., description="Background and expertise of the agent")

# New state management types for LangGraph
class CompanyProfileState(BaseModel):
    vision: str = Field(default="", description="Company vision and mission")
    technical_focus: str = Field(default="", description="Company's technical focus areas")
    website: str = Field(default="", description="Company website URL")
    sbir_experience: str = Field(default="", description="SBIR/STTR Program/Project Experience")
    innovations: str = Field(default="", description="Key innovations and capabilities")
    technical_experience: str = Field(default="", description="Technical experience and past performance")

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
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    messages: Sequence[BaseMessage] = Field(default_factory=list)
    company_profile: CompanyProfileState = Field(default_factory=CompanyProfileState)
    search_requirements: SearchRequirementsState = Field(default_factory=SearchRequirementsState)
    funding_sources: Dict[str, FundingSourceState] = Field(default_factory=dict)
    grant_opportunities: List[GrantOpportunityState] = Field(default_factory=list)
    final_report: Dict[str, Any] = Field(default_factory=dict)
    config: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    search_iterations: int = Field(default=0)
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    session_id: str = Field(default_factory=lambda: f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    chat_history: List[Dict[str, Any]] = Field(default_factory=list)
    searched_websites: set[str] = Field(default_factory=set)
    search_history: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Fix the search_progress type to use proper nested structure
    search_progress: Dict[str, Any] = Field(
        default_factory=lambda: {
            "sources_searched": [],
            "total_sources": 0,
            "successful_searches": 0,
            "failed_searches": 0
        }
    )
    
    # Strategy tracking
    strategic_plan: Dict[str, Any] = Field(default_factory=dict)
    identified_gaps: List[Dict[str, str]] = Field(default_factory=list)
    planned_actions: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Search refinement tracking
    refinement_history: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Validation tracking
    validation_results: Dict[str, List[str]] = Field(default_factory=lambda: {
        "errors": [],
        "warnings": [],
        "suggestions": []
    })

    def add_to_history(self, action: str, data: Any) -> None:
        """Add an action and its data to search history with timestamp"""
        self.search_history.append({
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "data": data
        })

    def track_search(self, source: str, success: bool, error: Optional[str] = None):
        """Track individual source search results"""
        self.search_progress["sources_searched"].append(source)
        if success:
            self.search_progress["successful_searches"] += 1
        else:
            self.search_progress["failed_searches"] += 1
            if error:
                self.validation_results["errors"].append(f"{source}: {error}")
        # Add to general history
        self.add_to_history("search_attempt", {
            "source": source,
            "success": success,
            "error": error
        })

    def add_gap(self, gap_type: str, description: str):
        """Track identified information gaps"""
        gap_entry = {
            "type": gap_type,
            "description": description,
            "timestamp": datetime.now().isoformat()
        }
        self.identified_gaps.append(gap_entry)
        # Add to general history
        self.add_to_history("gap_identified", gap_entry)

    def add_planned_action(self, action: Dict[str, Any]):
        """Track planned strategic actions"""
        action_entry = {
            **action,
            "status": "pending",
            "created_at": datetime.now().isoformat()
        }
        self.planned_actions.append(action_entry)
        # Add to general history
        self.add_to_history("action_planned", action_entry)

    def update_search_refinement(self, refinement_data: Dict[str, Any]):
        """Track search refinement history"""
        refinement_entry = {
            **refinement_data,
            "iteration": self.search_iterations,
            "timestamp": datetime.now().isoformat()
        }
        self.refinement_history.append(refinement_entry)
        # Add to general history
        self.add_to_history("search_refined", refinement_entry)

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
class ProfileAnalysisInput(BaseModel):
    company_documents: List[str]
    config: Dict[str, Any]

class ProfileAnalysisOutput(BaseModel):
    company_profile: CompanyProfileState
    errors: List[str] = Field(default_factory=list)

class StrategyDevelopmentInput(BaseModel):
    company_profile: CompanyProfileState
    company_focus: str
    organization_focus: str

class StrategyDevelopmentOutput(BaseModel):
    search_requirements: SearchRequirementsState
    errors: List[str] = Field(default_factory=list)

class GrantSearchInput(BaseModel):
    search_requirements: SearchRequirementsState
    company_profile: CompanyProfileState
    funding_sources: Dict[str, Any]
    search_iterations: int = 0

class GrantSearchOutput(BaseModel):
    grant_opportunities: List[GrantOpportunityState]
    funding_sources: Dict[str, FundingSourceState]
    errors: List[str] = Field(default_factory=list)

class QualityCheckInput(BaseModel):
    grant_opportunities: List[GrantOpportunityState]
    search_requirements: SearchRequirementsState
    company_profile: CompanyProfileState
    search_iterations: int = 0

class QualityCheckOutput(BaseModel):
    quality_met: bool
    refinements: Optional[Dict[str, Any]] = None
    errors: List[str] = Field(default_factory=list)

class FinalReportInput(BaseModel):
    company_profile: CompanyProfileState
    search_requirements: SearchRequirementsState
    grant_opportunities: List[GrantOpportunityState]
    company_focus: str
    organization_focus: str

class FinalReportOutput(BaseModel):
    final_report: Dict[str, Any]
    errors: List[str] = Field(default_factory=list)
