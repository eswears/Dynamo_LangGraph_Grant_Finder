# nodes.py
from typing import Annotated, Dict, List, Optional, Any, Union, Tuple
from pydantic import ValidationError
import json
from langchain_core.messages import BaseMessage, FunctionMessage
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.tools import Tool, BaseTool  
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import START, END, StateGraph
from datetime import datetime
import logging
from pathlib import Path
import yaml
from typing_extensions import TypeVar
import time
import os
from openai import RateLimitError, OpenAIError
from .models.base import BaseLanguageModel

from .tools import (
    CompanyDocumentTool,
    EnhancedGrantSearchTool,
    WebScrapeTool,
    FundingSourceTool,
    StrategicPlannerTool,
    StrategyRequirementsTool,  
    FinalReportTool
)

from .types import (
    GrantFinderState, CompanyProfileState, SearchRequirementsState, 
    GrantOpportunityState, FundingSourceState,
    # Add new state schemas
    ProfileAnalysisInput, ProfileAnalysisOutput,
    StrategyDevelopmentInput, StrategyDevelopmentOutput,
    GrantSearchInput, GrantSearchOutput,
    QualityCheckInput, QualityCheckOutput,
    FinalReportInput, FinalReportOutput
)

def load_agent_config() -> Dict[str, any]:
    """Load agent configurations from YAML"""
    config_path = Path(__file__).parent / 'config' / 'agents.yaml'
    with open(config_path) as f:
        return yaml.safe_load(f)['agents']

def load_task_config() -> Dict[str, any]:
    """Load task configurations from YAML"""
    config_path = Path(__file__).parent / 'config' / 'tasks.yaml'
    with open(config_path) as f:
        return yaml.safe_load(f)['tasks']

class ProfileAnalysisNode:
    def __init__(self, company_context_path: str, logger: logging.Logger, llm: BaseLanguageModel):
        self.logger = logger
        self.llm = llm
        self.company_context_path = company_context_path
        # Initialize tool with all required fields
        self.tool = CompanyDocumentTool(
            directory_path=Path(company_context_path),
            logger=logger,
            llm=llm,
            config={
                "embeddings": {
                    "provider": "bitnet",
                    "save_path": str(Path(company_context_path) / "embeddings"),
                    "bitnet": {
                        "dimension": 1536,
                        "batch_size": 32
                    }
                }
            }
        )
        
        # Load agent and task configs - store for use in __call__
        self.agent_config = load_agent_config()['company_profiler']
        self.task_config = load_task_config()['extract_company_profile_task']
        
        self.logger.info("Initializing ProfileAnalysisNode with configs:")
        self.logger.info(f"Agent config: {json.dumps(self.agent_config, indent=2)}")
        self.logger.info(f"Task config: {json.dumps(self.task_config, indent=2)}")
    
    def __call__(self, state: GrantFinderState) -> GrantFinderState:
        try:
            self.logger.info("Starting company profile analysis")
            input_data = {
                "profile_data": state.company_profile.dict(),
                "company_focus": state.company_focus,
                "config": state.config
            }
            try:
                doc_results = self.tool._run(json.dumps(input_data))
            except Exception as e:
                state.errors.append(f"Document analysis failed: {str(e)}")
                # Provide minimal valid results to allow workflow to continue
                doc_results = {
                    "high_level": {"summary": []},
                    "mid_level": {"details": []},
                    "low_level": {"specifics": []}
                }
            
            # Use high-level summary for vision and overall focus
            high_level = doc_results["high_level"]["summary"]
            state.company_profile.vision = high_level[0]["content"] if high_level else ""
            
            # Use mid-level details for technical capabilities
            mid_level = doc_results["mid_level"]["details"]
            state.company_profile.technical_focus = "\n".join(
                detail["content"] for detail in mid_level
            )
            
            # Use specific examples for experience and innovations
            low_level = doc_results["low_level"]["specifics"]
            state.company_profile.technical_experience = "\n".join(
                example["content"] for example in low_level
                if "experience" in example["content"].lower()
            )
            
            state.company_profile.innovations = "\n".join(
                example["content"] for example in low_level
                if "innovation" in example["content"].lower()
            )
            
            return state
            
        except Exception as e:
            self.logger.error(f"Profile analysis failed: {str(e)}")
            state.errors.append(f"Profile analysis error: {str(e)}")
            return state

class StrategyDevelopmentNode:
    def __init__(self, logger: logging.Logger, llm: BaseLanguageModel):
        self.logger = logger
        self.llm = llm  # Store LLM for later use
        
        # Create properly structured tool
        self.tool = StrategyRequirementsTool()
        
        # Load agent and task configs - store for use in __call__
        self.agent_config = load_agent_config()['strategic_writer_agent']
        self.task_config = load_task_config()['write_strategic_requirements_task']
        
        self.logger.info("Initializing StrategyDevelopmentNode with configs:")
        self.logger.info(f"Agent config: {json.dumps(self.agent_config, indent=2)}")
        self.logger.info(f"Task config: {json.dumps(self.task_config, indent=2)}")
    
    def __call__(self, state: GrantFinderState) -> GrantFinderState:
        try:
            self.logger.info("Starting strategy development")
            
            # Get hierarchical analysis with context
            doc_results = self.tool._run("Analyze strategic capabilities and requirements")
            
            # Use high-level summary for overall strategy
            high_level = doc_results["high_level"]
            strategic_focus = high_level["summary"][0]["content"]
            
            # Use mid-level details for specific requirements
            mid_level = doc_results["mid_level"]
            technical_requirements = [
                detail["content"] 
                for detail in mid_level["details"]
                if "requirement" in detail["content"].lower()
            ]
            
            # Use cross-references to maintain context
            state.search_requirements.technical_requirements = technical_requirements
            state.search_requirements.context = {
                "high_level": high_level["context"],
                "mid_level": mid_level["context"]
            }
            
            return state
            
        except Exception as e:
            self.logger.error(f"Strategy development failed: {str(e)}")
            state.errors.append(f"Strategy development error: {str(e)}")
            return state

class StrategicPlannerNode:
    def __init__(self, logger: logging.Logger, llm: BaseLanguageModel):
        self.logger = logger
        self.llm = llm
        self.tool = StrategicPlannerTool()
        
        # Load agent and task configs
        self.agent_config = load_agent_config()['strategic_planner_agent']
        self.task_config = load_task_config()['strategic_planning_task']        
    
    def __call__(self, state: GrantFinderState) -> GrantFinderState:
        try:
            self.logger.info("Starting strategic planning")
            
            # Format task description and output with state variables
            self.task_config['description'] = self.task_config['description'].replace("{company_focus}", f"{state.config['company_focus']}")
            self.task_config['expected_output'] = self.task_config['expected_output'].replace("{company_focus}", f"{state.config['company_focus']}")
            
            # Define tools information properly
            tools = [self.tool]
            tool_names = [tool.name for tool in tools]
            tools_desc = "\n".join(f"- {tool.name}: {tool.description}" for tool in tools)
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", f"""You are {self.agent_config['name']}, {self.agent_config['role']}.
                Background: {self.agent_config['backstory']}
                Goal: {self.agent_config['goal']}
                
                Task: {self.task_config['description']}
                Expected Output: {self.task_config['expected_output']}"""),
                ("user", "{input}"),
                ("assistant", "{agent_scratchpad}"),
                ("system", "Available tools:\n{tools}")
            ])
            
            agent = create_structured_chat_agent(
                llm=self.llm,
                tools=tools,
                prompt=prompt.partial(
                    tools=tools_desc,
                    tool_names=tool_names
                )
            )
            executor = AgentExecutor(
                agent=agent,
                tools=tools,
                handle_parsing_errors=True,
                max_iterations=3
            )

            self.logger.info(f"Task config: {json.dumps(self.task_config, indent=2)}")
            
            # Prepare input data
            planning_data = {
                "opportunities": state.grant_opportunities,
                "company_profile": state.company_profile.dict(),
                "search_requirements": state.search_requirements.dict(),
                "events": state.events if hasattr(state, "events") else []
            }
            
            # Generate strategic plan
            result = executor.invoke({
                "input": f"Develop strategic plan using: {json.dumps(planning_data)}"
            })
            
            # Check for information gaps
            if result.get("information_gaps"):
                # Increment search iterations to fill gaps
                state.search_iterations += 1
                state.search_requirements.technical_requirements.extend(
                    gap.replace("Missing capability: ", "") 
                    for gap in result["information_gaps"]
                )
            
            # Store strategic plan
            state.strategic_plan = result
            
            return state
            
        except Exception as e:
            self.logger.error(f"Strategic planning failed: {str(e)}")
            state.errors.append(f"Strategic planning error: {str(e)}")
            return state

class GrantSearchNode:
    def __init__(self, funding_sources_path: str, logger: logging.Logger, llm: BaseLanguageModel, api_key: str):
        self.logger = logger
        self.llm = llm
        self.max_retries = 3
        self.retry_delay = 1  # Initial delay in seconds
        
        self.tools = [
            EnhancedGrantSearchTool(
                serp_api_key=api_key,
                funding_sources_path=funding_sources_path,
                logger=logger
            ),
            WebScrapeTool(logger),
            FundingSourceTool(funding_sources_path, logger)
        ]
        
        # Load agent and task configs
        self.agent_config = load_agent_config()['fed_grant_search_agent']
        self.task_config = load_task_config()['fed_grant_task']
        
    def _search_with_backoff(self, source_name: str, source_data: Dict, state: GrantFinderState) -> Tuple[bool, List[GrantOpportunityState]]:
        """Perform search with exponential backoff"""
        opportunities = []
        
        # Define tools information properly
        tools = self.tools
        tool_names = [tool.name for tool in tools]
        tools_desc = "\n".join(f"- {tool.name}: {tool.description}" for tool in tools)
        
        # Create prompt template with state-dependent variables
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are {self.agent_config['name']}, {self.agent_config['role']}.
            Background: {self.agent_config['backstory']}
            Goal: {self.agent_config['goal']}
            
            Task: {self.task_config['description']}
            Expected Output: {self.task_config['expected_output']}"""),
            ("user", "{input}"),
            ("assistant", "{agent_scratchpad}"),
            ("system", "Available tools:\n{tools}")
        ])
    
        # Create agent and executor
        agent = create_structured_chat_agent(
            llm=self.llm,
            tools=tools,
            prompt=prompt.partial(
                tools=tools_desc,
                tool_names=tool_names
            )
        )
        executor = AgentExecutor(
            agent=agent, 
            tools=self.tools, 
            llm=self.llm,
            handle_parsing_errors=True,
            max_iterations=3
        )

        for attempt in range(self.max_retries):
            try:
                if attempt > 0:
                    delay = self.retry_delay * (2 ** attempt)
                    self.logger.info(f"Retrying {source_name} after {delay} seconds (attempt {attempt + 1})")
                    time.sleep(delay)
                
                # Create search context with serializable data
                search_context = {
                    "source_name": source_name,
                    "source_url": source_data["url"],
                    "requirements": state.search_requirements.dict(),  # Convert to dict
                    "company_focus": state.config["company_focus"],
                    "organization_focus": state.config["organization_focus"]
                }
                
                # Search for opportunities
                result = executor.invoke({
                    "input": f"Find grant opportunities from source using context: {json.dumps(search_context)}"
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
                    return True, opportunities
                    
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise e
                continue
        
        return False, opportunities

    def _retry_failed_sources(self, failed_sources: List[str], sources: Dict, state: GrantFinderState) -> List[GrantOpportunityState]:
        """Retry failed source searches"""
        retry_opportunities = []
        
        for source_name in failed_sources:
            try:
                self.logger.info(f"Retrying failed source: {source_name}")
                success, opportunities = self._search_with_backoff(source_name, sources[source_name], state)
                
                if success:
                    retry_opportunities.extend(opportunities)
                    sources[source_name].update({
                        "was_searched": True,
                        "search_successful": True,
                        "search_timestamp": datetime.now().isoformat(),
                        "grants_found": opportunities
                    })
                    state.track_search(source_name, True)
                else:
                    sources[source_name].update({
                        "was_searched": True,
                        "search_successful": False,
                        "error_message": "Search failed after retries",
                        "search_timestamp": datetime.now().isoformat()
                    })
                    state.track_search(source_name, False, "Search failed after retries")
                    
            except Exception as e:
                self.logger.error(f"Retry failed for source {source_name}: {str(e)}")
                sources[source_name].update({
                    "was_searched": True,
                    "search_successful": False,
                    "error_message": str(e),
                    "search_timestamp": datetime.now().isoformat()
                })
                state.track_search(source_name, False, str(e))
        
        return retry_opportunities

    def __call__(self, state: GrantFinderState) -> GrantFinderState:
        try:
            self.logger.info("Starting grant opportunity search")
            
            # Format task description and output with state variables
            self.task_config['description'] = self.task_config['description'].replace("{company_focus}", f"{state.config['company_focus']}")
            self.task_config['expected_output'] = self.task_config['expected_output'].replace("{company_focus}", f"{state.config['company_focus']}")
            
            self.logger.info(f"Task config: {json.dumps(self.task_config, indent=2)}")
            
            if not hasattr(state, 'config'):
                state.config = {}

            state.config['company_focus'] = state.company_focus
            state.config['organization_focus'] = state.grant_org

            input_data = GrantSearchInput(
                search_requirements=state.search_requirements,
                company_profile=state.company_profile,
                funding_sources=state.funding_sources,
                search_iterations=state.search_iterations
            )
            # Load funding sources
            funding_tool = next(t for t in self.tools if isinstance(t, FundingSourceTool))
            sources = funding_tool._run("load")
            
            opportunities: List[GrantOpportunityState] = []
            failed_sources: List[str] = []
            
            # Process each funding source
            for source_name, source_data in sources.items():
                try:
                    self.logger.info(f"Processing source: {source_name}")
                    success, source_opportunities = self._search_with_backoff(source_name, source_data, state)
                    
                    if success:
                        opportunities.extend(source_opportunities)
                        source_data.update({
                            "was_searched": True,
                            "search_successful": True,
                            "search_timestamp": datetime.now().isoformat(),
                            "grants_found": source_opportunities
                        })
                        state.track_search(source_name, True)
                    else:
                        failed_sources.append(source_name)
                        source_data.update({
                            "was_searched": True,
                            "search_successful": False,
                            "error_message": "Initial search failed",
                            "search_timestamp": datetime.now().isoformat()
                        })
                        state.track_search(source_name, False, "Initial search failed")
                    
                except Exception as e:
                    self.logger.error(f"Error processing source {source_name}: {str(e)}")
                    failed_sources.append(source_name)
                    source_data.update({
                        "was_searched": True,
                        "search_successful": False,
                        "error_message": str(e),
                        "search_timestamp": datetime.now().isoformat()
                    })
                    state.track_search(source_name, False, str(e))
            
            # Retry failed sources
            if failed_sources:
                self.logger.info(f"Retrying {len(failed_sources)} failed sources")
                retry_opportunities = self._retry_failed_sources(failed_sources, sources, state)
                opportunities.extend(retry_opportunities)
            
            # Validate output
            output_data = GrantSearchOutput(
                grant_opportunities=opportunities,
                funding_sources=sources,
                errors=[]
            )
            
            state.funding_sources = output_data.funding_sources
            state.grant_opportunities = output_data.grant_opportunities
            
            # Log summary
            self.logger.info(f"Search completed - Successful: {len(sources) - len(failed_sources)}, "
                           f"Failed: {len(failed_sources)}, Total opportunities: {len(opportunities)}")
            
            return state
            
        except ValidationError as e:
            self.logger.error(f"Grant search validation failed: {str(e)}")
            state.errors.append(f"Grant search validation error: {str(e)}")
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
        
class QualityCheckNode:
    def __init__(self, logger: logging.Logger, llm: BaseLanguageModel):
        self.logger = logger
        self.llm = llm
        
        # Load agent and task configs - store for use in __call__
        self.agent_config = load_agent_config()['quality_check_agent']
        self.task_config = load_task_config()['quality_check_task']
        
        self.logger.info("Initializing QualityCheckNode with configs:")
        self.logger.info(f"Agent config: {json.dumps(self.agent_config, indent=2)}")
    
    def __call__(self, state: GrantFinderState) -> GrantFinderState:
        try:
            self.logger.info("Evaluating search result quality")
            
            # Basic quality check first
            if state.search_iterations < 3 and len(state.grant_opportunities) < 5:
                state.search_iterations += 1
                return state
            
            # Format task description and output with state variables
            self.task_config['description'] = self.task_config['description'].replace("{company_focus}", f"{state.config['company_focus']}")
            self.task_config['expected_output'] = self.task_config['expected_output'].replace("{company_focus}", f"{state.config['company_focus']}")
            
            # Create prompt template with state-dependent variables
            prompt = ChatPromptTemplate.from_messages([
                ("system", f"""You are {self.agent_config['name']}, {self.agent_config['role']}.
                Background: {self.agent_config['backstory']}
                Goal: {self.agent_config['goal']}
                
                Task: {self.task_config['description']}
                Expected Output: {self.task_config['expected_output']}"""),
                ("user", "{input}"),
                ("assistant", "{agent_scratchpad}")
            ])
            
            self.logger.info(f"Task config: {json.dumps(self.task_config, indent=2)}")

            # Create agent with current prompt
            agent = create_structured_chat_agent(self.llm, [], prompt)
            executor = AgentExecutor(agent=agent, tools=[])
            
            # Add executor.invoke call
            result = executor.invoke({
                "input": json.dumps({
                    "opportunities": [opp.model_dump() for opp in state.grant_opportunities],
                    "requirements": state.search_requirements.model_dump(),
                    "iterations": state.search_iterations
                }),
                "agent_scratchpad": ""
            })
            
            # Save results to state
            state.validation_results["quality_check"] = result
            state.add_to_history("quality_check", result)
            
            # If gaps were identified, add them to state
            if "identified_gaps" in result:
                for gap in result["identified_gaps"]:
                    state.add_gap("quality", gap)
            
            return state
            
        except Exception as e:
            self.logger.error(f"Quality check failed: {str(e)}")
            state.errors.append(f"Quality check error: {str(e)}")
            return state

class FinalReportNode:
    def __init__(self, logger: logging.Logger, llm: BaseLanguageModel, output_dir: Path):
        self.logger = logger
        self.llm = llm
        self.output_dir = output_dir
        self.tool = FinalReportTool()
        
        # Load agent and task configs - store for use in __call__
        self.agent_config = load_agent_config()['fed_grant_report_agent']
        self.task_config = load_task_config()['federal_grant_report_task']
        
        self.logger.info("Initializing FinalReportNode with configs:")
        self.logger.info(f"Agent config: {json.dumps(self.agent_config, indent=2)}")
    
    def __call__(self, state: GrantFinderState) -> GrantFinderState:
        try:
            self.logger.info("Starting final report generation")
            self.task_config['description'] = self.task_config['description'].replace("{company_focus}", f"{state.config['company_focus']}")
            self.task_config['expected_output'] = self.task_config['expected_output'].replace("{company_focus}", f"{state.config['company_focus']}")
            
            # Create prompt template with state-dependent variables
            prompt = ChatPromptTemplate.from_messages([
                ("system", f"""You are {self.agent_config['name']}, {self.agent_config['role']}.
                Background: {self.agent_config['backstory']}
                Goal: {self.agent_config['goal']}
                
                Task: {self.task_config['description']}
                Expected Output: {self.task_config['expected_output']}"""),
                ("user", "{input}"),
                ("assistant", "{agent_scratchpad}")
            ])
            self.logger.info(f"Task config: {json.dumps(self.task_config, indent=2)}")

            # Create agent with current prompt
            agent = create_structured_chat_agent(self.llm, [self.tool], prompt)
            self.executor = AgentExecutor(agent=agent, tools=[self.tool])
            
            # Validate input state
            input_data = FinalReportInput(
                company_profile=state.company_profile,
                search_requirements=state.search_requirements,
                grant_opportunities=state.grant_opportunities,
                company_focus=state.config["company_focus"],
                organization_focus=state.config["organization_focus"]
            )
            
            result = self.executor.invoke({
                "input": f"Create comprehensive grant analysis report using context: {input_data.dict()}"
            })
            
            # Validate output
            output_data = FinalReportOutput(
                final_report={
                    "timestamp": datetime.now().isoformat(),
                    "executive_summary": result.get("executive_summary", ""),
                    "opportunities": result.get("opportunities", []),
                    "strategic_recommendations": result.get("strategic_recommendations", []),
                    "action_plan": result.get("action_plan", [])
                }
            )
            
            # Save report
            report_path = self.output_dir / f"grant_report_{state.timestamp}.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(report_path, 'w') as f:
                json.dump(output_data.final_report, f, indent=2)
            
            state.final_report = output_data.final_report
            return state
            
        except ValidationError as e:
            self.logger.error(f"Report generation validation failed: {str(e)}")
            state.errors.append(f"Report generation validation error: {str(e)}")
            return state
        except Exception as e:
            self.logger.error(f"Report generation failed: {str(e)}")
            state.errors.append(f"Report generation error: {str(e)}")
            return state

def build_graph(
    company_context_path: str,
    funding_sources_path: str,
    output_dir: Path,
    logger: logging.Logger,
    llm: BaseLanguageModel,
    serp_api_key: str
) -> StateGraph:
    """Builds the complete LangGraph workflow"""
    
    # Initialize nodes
    profile_node = ProfileAnalysisNode(company_context_path, logger, llm)
    strategy_node = StrategyDevelopmentNode(logger, llm)
    search_node = GrantSearchNode(funding_sources_path, logger, llm, serp_api_key)
    quality_node = QualityCheckNode(logger, llm)
    strategic_node = StrategicPlannerNode(logger, llm)
    report_node = FinalReportNode(logger, llm, output_dir)
    
    # Create graph
    workflow = StateGraph(GrantFinderState)
    
    # Add nodes
    workflow.add_node("analyze_profile", profile_node)
    workflow.add_node("develop_strategy", strategy_node)
    workflow.add_node("search_grants", search_node)
    workflow.add_node("check_quality", quality_node)
    workflow.add_node("plan_strategy", strategic_node)
    workflow.add_node("generate_report", report_node)
    
    # Add edges starting from START
    workflow.add_edge(START, "analyze_profile")
    workflow.add_edge("analyze_profile", "develop_strategy")
    workflow.add_edge("develop_strategy", "search_grants")
    workflow.add_edge("search_grants", "check_quality")
    
    # Define the conditional routing
    def route_by_quality(state: GrantFinderState) -> str:
        """Route based on search quality"""
        # Return True if we need more searching
        return "search_grants" if state.search_iterations < 3 and len(state.grant_opportunities) < 5 else "plan_strategy"

    def route_by_gaps(state: GrantFinderState) -> str:
        """Route based on information gaps"""
        return "search_grants" if state.strategic_plan.get("information_gaps") else "generate_report"

    workflow.add_conditional_edges(
        "check_quality",
        route_by_quality,
        {"search_grants": "search_grants", "plan_strategy": "plan_strategy"}
    )

    workflow.add_conditional_edges(
        "plan_strategy",
        route_by_gaps,
        {"search_grants": "search_grants", "generate_report": "generate_report"}
    )
    
    # Add final edge to END
    workflow.add_edge("generate_report", END)
    
    # Use set_finish_point instead of set_finish
    workflow.set_finish_point("generate_report")
    
    return workflow