# main.py
from typing import Dict, Any, TypeVar
from pathlib import Path
import logging
import yaml
import os
import json
from datetime import datetime
from dotenv import load_dotenv
from .persistence import PersistenceManager
from .utils.config import get_llm_config
from langgraph.graph import Graph, StateGraph
from grant_finder.models.base import BaseLanguageModel
from grant_finder.models.bitnet import BitNetLLM
from grant_finder.models.openai_wrapper import OpenAIWrapper

from .types import (
    GrantFinderState, LogConfig, OutputConfig, 
    UserInputConfig, GrantSearchError
)
from .nodes import build_graph

# Initialize logger at module level
logger = logging.getLogger('grant_finder')

State = TypeVar("State", bound=GrantFinderState)

def setup_logging(timestamp: str, output_dir: Path) -> LogConfig:
    """Initialize logging configuration"""
    # Create log directory
    log_dir = output_dir / timestamp / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create formatters
    file_formatter = logging.Formatter(
        '[%(asctime)s] %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(message)s'
    )
    
    # Setup system logger
    system_logger = logging.getLogger('grant_finder')
    system_logger.setLevel(logging.INFO)
    
    system_file_handler = logging.FileHandler(
        log_dir / "grant_finder.log",
        encoding='utf-8'
    )
    system_file_handler.setFormatter(file_formatter)
    
    system_console_handler = logging.StreamHandler()
    system_console_handler.setFormatter(console_formatter)
    
    system_logger.addHandler(system_file_handler)
    system_logger.addHandler(system_console_handler)
    
    # Setup agent logger
    agent_logger = logging.getLogger('grant_finder.agents')
    agent_logger.setLevel(logging.INFO)
    
    agent_file_handler = logging.FileHandler(
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

def main():
    try:
        print("Starting Grant Finder process")                

        # Load configuration
        config = load_config("config/user_config.yaml")
        llm_config = get_llm_config(config)

        # Create default output dir for logging
        output_dir = Path(config["output"]["output_directory"])
        output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_config = setup_logging(timestamp, output_dir)
        logger = log_config["system_logger"]

        # Load environment variables from the correct location
        env_path = Path(__file__).parent / '.env'
        if not env_path.exists():
            # Try one directory up if not found
            env_path = Path(__file__).parent.parent / '.env'
        load_dotenv(env_path)
        
        # Initialize LLM based on type
        model_type = config.get("llm", {}).get("model_type", "openai")
        
        if model_type == "openai":
            # Get API key for OpenAI
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                raise GrantSearchError("OPENAI_API_KEY environment variable not found")
            llm = OpenAIWrapper(**llm_config, api_key=openai_api_key)
        elif model_type == "bitnet":
            llm = BitNetLLM(**llm_config)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        logger.info(f"Initialized {model_type} model")
        
        # Get SERP API key for search functionality
        serp_api_key = os.getenv("SERPER_API_KEY") or os.getenv("SERP_API_KEY")
        if not serp_api_key:
            raise GrantSearchError("Neither SERPER_API_KEY nor SERP_API_KEY environment variable found")

        # Initialize persistence manager
        persistence = PersistenceManager(output_dir / "persistence")
        
        # Try to load previous session
        previous_state = persistence.load_last_session()
        if previous_state:
            logger.info("Found previous session")
            print("\nFound previous session. Would you like to:")
            print("1. Continue previous search")
            print("2. Start new search")
            choice = input("Enter choice (1 or 2): ").strip()
            
            if choice == "1":
                logger.info("Resuming previous session")
                initial_state = GrantFinderState(**previous_state)
                print("\nPrevious searches:", ", ".join(s["action"] for s in initial_state.search_history))
                print("Would you like to:")
                print("1. Expand current search areas")
                print("2. Search new areas")
                expand_choice = input("Enter choice (1 or 2): ").strip()
                
                company_focus = (
                    initial_state.config["company_focus"] if expand_choice == "1"
                    else get_user_input(
                        "What is the company's main focus for pursuing SBIR/STTR grants?",
                        "Artificial Intelligence and Machine Learning"
                    )
                )
                
                organization_focus = (
                    initial_state.config["organization_focus"] if expand_choice == "1"
                    else get_user_input(
                        "What Grant organization are you interested in (DoD, EPA, Army, Air Force, etc.)?",
                        "DoD"
                    )
                )
            else:
                company_focus = get_user_input(
                    "What is the company's main focus for pursuing SBIR/STTR grants?",
                    "Artificial Intelligence and Machine Learning"
                )
                
                organization_focus = get_user_input(
                    "What Grant organization are you interested in (DoD, EPA, Army, Air Force, etc.)?",
                    "DoD"
                )
        else:
            company_focus = get_user_input(
                "What is the company's main focus for pursuing SBIR/STTR grants?",
                "Artificial Intelligence and Machine Learning"
            )
            
            organization_focus = get_user_input(
                "What Grant organization are you interested in (DoD, EPA, Army, Air Force, etc.)?",
                "DoD"
            )

        # Build workflow using StateGraph instead of just Graph
        workflow = StateGraph(State)
        
        # Add nodes using build_graph
        nodes = build_graph(
            company_context_path=config["company_context"]["directory"],
            funding_sources_path=config["funding_sources"]["file_path"],
            output_dir=output_dir,
            logger=logger,
            llm=llm,
            serp_api_key=serp_api_key
        )
        
        # Initialize state
        initial_state = GrantFinderState(
            timestamp=timestamp,
            config={
                "company_focus": company_focus,
                "organization_focus": organization_focus,
                "company_context_path": config["company_context"]["directory"],
                "funding_sources_path": config["funding_sources"]["file_path"]
            }
        )
        
        # Compile and execute workflow with proper typing
        app = nodes.compile()
        final_state = app.invoke(initial_state)
        
        # Save results
        save_results(final_state, output_dir, timestamp)
        
        # Save state for persistence
        persistence.save_state(final_state)
        
        logger.info("Grant Finder process completed successfully")
        
    except Exception as e:
        logger.error(f"Process failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()