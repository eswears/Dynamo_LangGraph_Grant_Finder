# nodes.py
from typing import Annotated, Dict, List
import json
from langchain_core.messages import BaseMessage, FunctionMessage
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from datetime import datetime
import logging
from pathlib import Path

from .types import (
    GrantFinderState, CompanyProfileState, 
    SearchRequirementsState, GrantOpportunityState
)
from .tools import (
    CompanyDocumentTool, GrantSearchTool, 
    WebScrapeTool, FundingSourceTool
)

class ProfileAnalysisNode:
    def __init__(
        self,
        company_context_path: str,
        logger: logging.Logger,
        llm: ChatOpenAI
    ):
        self.logger = logger
        self.tool = CompanyDocumentTool(company_context_path, logger)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at analyzing company documents to extract detailed profiles."),
            ("user", "{input}")
        ])
        
        agent = create_openai_tools_agent(llm, [self.tool], prompt)
        self.executor = AgentExecutor(agent=agent, tools=[self.tool])
    
    def __call__(self, state: GrantFinderState) -> GrantFinderState:
        try:
            self.logger.info("Starting company profile analysis")
            
            result = self.executor.invoke({
                "input": "Analyze company documents to extract complete profile"
            })
            
            # Update state with profile information
            state.company_profile = CompanyProfileState(
                vision=result.get("vision", ""),
                technical_focus=result.get("technical_focus", ""),
                website=result.get("website", ""),
                sbir_experience=result.get("sbir_experience", ""),
                innovations=result.get("innovations", ""),
                technical_experience=result.get("technical_experience", "")
            )
            
            return state
            
        except Exception as e:
            self.logger.error(f"Profile analysis failed: {str(e)}")
            state.errors.append(f"Profile analysis error: {str(e)}")
            return state

class StrategyDevelopmentNode:
    def __init__(
        self,
        logger: logging.Logger,
        llm: ChatOpenAI
    ):
        self.logger = logger
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at developing strategic search requirements based on company profiles."),
            ("user", "{input}")
        ])
        
        agent = create_openai_tools_agent(llm, [], prompt)
        self.executor = AgentExecutor(agent=agent, tools=[])
    
    def __call__(self, state: GrantFinderState) -> GrantFinderState:
        try:
            self.logger.info("Starting strategy development")
            
            # Create input context from company profile
            context = {
                "company_profile": state.company_profile,
                "company_focus": state.config["company_focus"]
            }
            
            result = self.executor.invoke({
                "input": f"Develop strategic search requirements based on: {context}"
            })
            
            # Update state with search requirements
            state.search_requirements = SearchRequirementsState(
                technical_requirements=result.get("technical_requirements", []),
                innovation_areas=result.get("innovation_areas", []),
                competitive_advantages=result.get("competitive_advantages", []),
                target_phases=result.get("target_phases", [])
            )
            
            return state
            
        except Exception as e:
            self.logger.error(f"Strategy development failed: {str(e)}")
            state.errors.append(f"Strategy development error: {str(e)}")
            return state

class GrantSearchNode:
    def __init__(
        self,
        funding_sources_path: str,
        logger: logging.Logger,
        llm: ChatOpenAI,
        api_key: str
    ):
        self.logger = logger
        self.tools = [
            GrantSearchTool(api_key, logger),
            WebScrapeTool(logger),
            FundingSourceTool(funding_sources_path, logger)
        ]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at finding and analyzing grant opportunities."),
            ("user", "{input}")
        ])
        
        agent = create_openai_tools_agent(llm, self.tools, prompt)
        self.executor = AgentExecutor(agent=agent, tools=self.tools)
    
    def __call__(self, state: GrantFinderState) -> GrantFinderState:
        try:
            self.logger.info("Starting grant opportunity search")
            
            # Load funding sources
            funding_tool = next(t for t in self.tools if isinstance(t, FundingSourceTool))
            sources = funding_tool._run("load")
            
            opportunities: List[GrantOpportunityState] = []
            
            # Process each funding source
            for source_name, source_data in sources.items():
                try:
                    self.logger.info(f"Processing source: {source_name}")
                    
                    # Create search context
                    search_context = {
                        "source_name": source_name,
                        "source_url": source_data["url"],
                        "requirements": state.search_requirements,
                        "company_focus": state.config["company_focus"],
                        "organization_focus": state.config["organization_focus"]
                    }
                    
                    # Search for opportunities
                    result = self.executor.invoke({
                        "input": f"Find grant opportunities from source using context: {search_context}"
                    })
                    
                    # Process found opportunities
                    if isinstance(result, dict) and "opportunities" in result:
                        for opp in result["opportunities"]:
                            if self._validate_opportunity(opp):
                                opportunities.append(GrantOpportunityState(
                                    topic_id=opp["topic_id"],
                                    title=opp["title"],
                                    description=opp["description"],
                                    technical_requirements=opp["technical_requirements"],
                                    alignment_score=float(opp["alignment_score"]),
                                    award_amount=opp["award_amount"],
                                    deadline=opp["deadline"],
                                    focus_areas=opp["focus_areas"],
                                    url=opp["url"],
                                    contact_info=opp["contact_info"],
                                    justification=opp["justification"]
                                ))
                    
                    # Update source status
                    source_data.update({
                        "was_searched": True,
                        "search_successful": True,
                        "search_timestamp": datetime.now().isoformat(),
                        "grants_found": opportunities
                    })
                    
                except Exception as e:
                    self.logger.error(f"Error processing source {source_name}: {str(e)}")
                    source_data.update({
                        "was_searched": True,
                        "search_successful": False,
                        "error_message": str(e),
                        "search_timestamp": datetime.now().isoformat()
                    })
            
            # Update state
            state.funding_sources = sources
            state.grant_opportunities = opportunities
            
            return state
            
        except Exception as e:
            self.logger.error(f"Grant search failed: {str(e)}")
            state.errors.append(f"Grant search error: {str(e)}")
            return state
    
    def _validate_opportunity(self, opp: Dict) -> bool:
        """Validates that an opportunity has all required fields and proper format"""
        required_fields = {
            "topic_id": str,
            "title": str,
            "description": str,
            "technical_requirements": list,
            "alignment_score": (int, float),
            "award_amount": str,
            "deadline": str,
            "focus_areas": list,
            "url": str
        }
        
        try:
            for field, field_type in required_fields.items():
                if field not in opp:
                    return False
                if not isinstance(opp[field], field_type):
                    return False
                
            # Additional validation rules
            if not opp["topic_id"].strip():
                return False
            if len(opp["description"]) < 100:  # Minimum description length
                return False
            if not opp["technical_requirements"]:
                return False
            if not any(char.isdigit() for char in opp["award_amount"]):
                return False
                
            return True
            
        except Exception:
            return False

class FinalReportNode:
    def __init__(
        self,
        logger: logging.Logger,
        llm: ChatOpenAI,
        output_dir: Path
    ):
        self.logger = logger
        self.output_dir = output_dir
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at analyzing grant opportunities and creating strategic reports."),
            ("user", "{input}")
        ])
        
        agent = create_openai_tools_agent(llm, [], prompt)
        self.executor = AgentExecutor(agent=agent, tools=[])
    
    def __call__(self, state: GrantFinderState) -> GrantFinderState:
        try:
            self.logger.info("Starting final report generation")
            
            # Create analysis context
            context = {
                "company_profile": state.company_profile,
                "search_requirements": state.search_requirements,
                "opportunities": state.grant_opportunities,
                "company_focus": state.config["company_focus"],
                "organization_focus": state.config["organization_focus"]
            }
            
            result = self.executor.invoke({
                "input": f"Create comprehensive grant analysis report using context: {context}"
            })
            
            # Format report
            report = {
                "timestamp": datetime.now().isoformat(),
                "executive_summary": result.get("executive_summary", ""),
                "opportunities": result.get("opportunities", []),
                "strategic_recommendations": result.get("strategic_recommendations", []),
                "action_plan": result.get("action_plan", [])
            }
            
            # Save report
            report_path = self.output_dir / f"grant_report_{state.timestamp}.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            self.logger.info(f"Saved final report to: {report_path}")
            
            # Update state
            state.final_report = report
            
            return state
            
        except Exception as e:
            self.logger.error(f"Report generation failed: {str(e)}")
            state.errors.append(f"Report generation error: {str(e)}")
            return state

class QualityCheckNode:
    """Evaluates search result quality and triggers refinement if needed"""
    
    def __init__(
        self,
        logger: logging.Logger,
        llm: ChatOpenAI,
        min_opportunities: int = 5,
        min_alignment_score: float = 70.0,
        max_iterations: int = 3
    ):
        self.logger = logger
        self.min_opportunities = min_opportunities
        self.min_alignment_score = min_alignment_score
        self.max_iterations = max_iterations
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at evaluating grant search results and identifying gaps in coverage."),
            ("user", "{input}")
        ])
        
        agent = create_openai_tools_agent(llm, [], prompt)
        self.executor = AgentExecutor(agent=agent, tools=[])
    
    def __call__(self, state: GrantFinderState) -> Tuple[str, GrantFinderState]:
        try:
            self.logger.info("Evaluating search result quality")
            
            # Check basic metrics
            if len(state.grant_opportunities) < self.min_opportunities:
                self.logger.info("Insufficient number of opportunities found")
                return "refine_search", state
                
            high_quality_opps = [
                opp for opp in state.grant_opportunities 
                if opp.alignment_score >= self.min_alignment_score
            ]
            
            if len(high_quality_opps) < self.min_opportunities:
                self.logger.info("Insufficient high-quality opportunities")
                return "refine_search", state
            
            # Analyze coverage and gaps
            analysis_input = {
                "opportunities": state.grant_opportunities,
                "search_requirements": state.search_requirements,
                "company_profile": state.company_profile,
                "iteration": state.get("search_iterations", 0)
            }
            
            result = self.executor.invoke({
                "input": f"Analyze search result coverage and identify gaps: {analysis_input}"
            })
            
            # Check if refinement needed
            if (result.get("has_gaps", True) and 
                state.get("search_iterations", 0) < self.max_iterations):
                
                # Update search requirements based on analysis
                state.search_requirements.update(result.get("refinements", {}))
                state.search_iterations = state.get("search_iterations", 0) + 1
                
                self.logger.info("Search refinement needed - updating requirements")
                return "refine_search", state
            
            self.logger.info("Search results meet quality criteria")
            return "complete", state
            
        except Exception as e:
            self.logger.error(f"Quality check failed: {str(e)}")
            state.errors.append(f"Quality check error: {str(e)}")
            return "complete", state

def build_graph(
    company_context_path: str,
    funding_sources_path: str,
    output_dir: Path,
    logger: logging.Logger,
    llm: ChatOpenAI,
    api_key: str
) -> StateGraph[GrantFinderState]:
    """Builds the complete LangGraph workflow"""
    
    # Initialize nodes
    profile_node = ProfileAnalysisNode(company_context_path, logger, llm)
    strategy_node = StrategyDevelopmentNode(logger, llm)
    search_node = GrantSearchNode(funding_sources_path, logger, llm, api_key)
    quality_node = QualityCheckNode(logger, llm)
    report_node = FinalReportNode(logger, llm, output_dir)
    
    # Create graph
    workflow = StateGraph(GrantFinderState)
    
    # Add nodes
    workflow.add_node("analyze_profile", profile_node)
    workflow.add_node("develop_strategy", strategy_node)
    workflow.add_node("search_grants", search_node)
    workflow.add_node("check_quality", quality_node)
    workflow.add_node("generate_report", report_node)
    
    # Add edges
    workflow.add_edge("analyze_profile", "develop_strategy")
    workflow.add_edge("develop_strategy", "search_grants")
    workflow.add_edge("search_grants", "check_quality")
    
    # Add conditional edges from quality check
    workflow.add_conditional_edges(
       "check_quality",
       {
           "refine_search": "search_grants",
           "complete": "generate_report"
       }
   )
    
    return workflow