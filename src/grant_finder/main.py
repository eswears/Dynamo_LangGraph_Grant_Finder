# main.py
from __future__ import annotations  # This should be first
from logging import getLogger, basicConfig, INFO, Formatter, FileHandler, StreamHandler  # Import specific logging functions
basicConfig(level=INFO)  # Configure logging

import os
import sys
from datetime import datetime  # Add datetime import
import yaml  # Add yaml import
from pathlib import Path
from typing import Dict, Any, Optional, TypeVar

from dotenv import load_dotenv
from langchain.agents import AgentExecutor
from langchain.tools import Tool
from langgraph.graph import StateGraph

# Load environment variables first
load_dotenv()

# Check for required environment variables
if not os.getenv("SERP_API_KEY"):
    raise ValueError("SERP_API_KEY environment variable is required but not set")

# Initialize logger at module level - MOVE THIS UP before using it
logger = getLogger('grant_finder')
logger.setLevel(INFO)
handler = StreamHandler()
handler.setFormatter(Formatter('%(message)s'))
logger.addHandler(handler)

from .nodes import (
    ProfileAnalysisNode,
    StrategyDevelopmentNode,
    QualityCheckNode,
    FinalReportNode,
    build_graph
)
from .utils.config import load_yaml_config, get_llm_config
from .utils.persistence import PersistenceManager
from .tools import (
    CompanyDocumentTool,
)
from .models.bitnet import BitNetLLM  # Add BitNetLLM import
from .models.openai_wrapper import OpenAIWrapper  # Add OpenAI import for completeness

from .types import (
    GrantFinderState, LogConfig, OutputConfig, 
    UserInputConfig, GrantSearchError
)

State = TypeVar("State", bound=GrantFinderState)

def setup_logging(timestamp: str, output_dir: Path) -> LogConfig:
    """Initialize logging configuration"""
    # Create log directory
    log_dir = output_dir / timestamp / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create formatters
    file_formatter = Formatter(
        '[%(asctime)s] %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = Formatter(
        '%(message)s'
    )
    
    # Setup system logger
    system_logger = getLogger('grant_finder')
    system_logger.setLevel(INFO)
    
    system_file_handler = FileHandler(
        log_dir / "grant_finder.log",
        encoding='utf-8'
    )
    system_file_handler.setFormatter(file_formatter)
    
    system_console_handler = StreamHandler()
    system_console_handler.setFormatter(console_formatter)
    
    system_logger.addHandler(system_file_handler)
    system_logger.addHandler(system_console_handler)
    
    # Setup agent logger
    agent_logger = getLogger('grant_finder.agents')
    agent_logger.setLevel(INFO)
    
    agent_file_handler = FileHandler(
        log_dir / "agent_interactions.log",
        encoding='utf-8'
    )
    agent_file_handler.setFormatter(file_formatter)
    agent_logger.addHandler(agent_file_handler)
    
    return LogConfig(
        level="INFO",
        file=str(log_dir / "grant_finder.log"),
        system_logger=system_logger,
        agent_logger=agent_logger
    )

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    try:
        # Use Path to handle paths correctly and find config relative to main.py
        config_file = Path(__file__).parent / 'config' / 'user_config.yaml'
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config: {str(e)}")
        raise

def get_user_input(prompt: str, default: str) -> str:
    """Get user input with default value"""
    print(f"\n{prompt}")
    print(f"[Press Enter to use default: {default}]")
    user_input = input().strip()
    return user_input if user_input else default

def save_results(state: GrantFinderState, output_dir: Path, timestamp: str) -> None:
    """Save final results to output directory"""
    results_dir = output_dir / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save final report
    if state.final_report:
        report_path = results_dir / "final_report.json"
        with open(report_path, 'w') as f:
            json.dump(state.final_report, f, indent=2)
    
    # Save opportunities
    if state.grant_opportunities:
        opps_path = results_dir / "opportunities.json"
        with open(opps_path, 'w') as f:
            json.dump([opp.dict() for opp in state.grant_opportunities], f, indent=2)
    
    # Save search history
    if state.search_history:
        history_path = results_dir / "search_history.json"
        with open(history_path, 'w') as f:
            json.dump(state.search_history, f, indent=2)

def main() -> None:
    """Main entry point for the application"""
    doc_store = None
    try:
        print("Starting Grant Finder process")
        
        # Load configuration
        config = load_config("config/user_config.yaml")
        
        # Initialize persistence manager
        persistence = PersistenceManager(Path(config['output']['output_directory']))
        
        # Create session ID
        session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Load previous context if exists
        previous_context = persistence.load_context(session_id)
        
        # Initialize language model based on config
        llm_config = get_llm_config(config)
        if config['llm']['model_type'] == 'bitnet':
            llm = BitNetLLM(**llm_config)
            logger.info("Initialized BitNet model")
        else:
            llm = OpenAIWrapper(**llm_config)
            logger.info("Initialized OpenAI model")
        
        # Initialize tools with context
        company_doc_tool = CompanyDocumentTool(
            directory_path=Path(config['company_context']['directory']),
            logger=logger,
            llm=llm,
            config=config,
            previous_context=previous_context
        )
        doc_store = company_doc_tool.doc_store
        
        # Get user input
        company_focus = get_user_input(
            "What is the company's main focus for pursuing SBIR/STTR grants?",
            "Artificial Intelligence and Machine Learning"
        )
        
        grant_org = get_user_input(
            "What Grant organization are you interested in (DoD, EPA, Army, Air Force, etc.)?",
            "DoD"
        )
        
        # Initialize nodes
        profile_node = ProfileAnalysisNode(
            company_context_path=str(Path(config['company_context']['directory'])),
            logger=logger,
            llm=llm
        )
        
        strategy_node = StrategyDevelopmentNode(
            logger=logger,
            llm=llm
        )
        
        quality_node = QualityCheckNode(
            logger=logger,
            llm=llm
        )
        
        report_node = FinalReportNode(
            logger=logger,
            llm=llm,
            output_dir=Path(config['output']['output_directory'])
        )
        
        # Build workflow graph
        workflow = build_graph(
            company_context_path=str(Path(config['company_context']['directory'])),
            funding_sources_path=str(Path(config['funding_sources']['file_path'])),
            output_dir=Path(config['output']['output_directory']),
            logger=logger,
            llm=llm,
            serp_api_key=os.getenv("SERP_API_KEY")
        )
        
        # Create initial state with context
        initial_state = GrantFinderState(
            company_focus=company_focus,
            grant_org=grant_org,
            output_dir=Path(config['output']['output_directory']),
            timestamp=session_id,
            context_history=previous_context.get('context_history', {}) if previous_context else {},
            layer_states=previous_context.get('layer_states', {}) if previous_context else {},
            config={
                'company_focus': company_focus,
                'organization_focus': grant_org,
                'company_context': config['company_context'],
                'funding_sources': config['funding_sources'],
                'output': config['output'],
                'embeddings': config['embeddings']
            },
            funding_sources={},  # Will be populated during search
            search_iterations=0,
            search_progress={
                "sources_searched": [],
                "total_sources": 0,
                "successful_searches": 0,
                "failed_searches": 0
            },
            errors=[],
            validation_results={
                "errors": [],
                "warnings": [],
                "suggestions": []
            }
        )
        
        # Compile the workflow before invoking
        app = workflow.compile()
        
        # Run workflow with compiled app
        final_state = app.invoke(initial_state)
        
        # Save context for future use
        persistence.save_context(session_id, {
            "context_history": final_state.context_history,
            "layer_states": final_state.layer_states
        })
        
        logger.info("Grant Finder process completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Process failed: {str(e)}")
        return 1
    finally:
        # Clean up resources
        if doc_store is not None:
            try:
                doc_store.cleanup()
            except Exception as e:
                logger.error(f"Error during cleanup: {str(e)}")

if __name__ == "__main__":
    sys.exit(main())