# main.py
from typing import Dict, Any
from pathlib import Path
import logging
import yaml
import os
import json
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from .persistence import PersistenceManager
from .utils.config import get_llm_config

from .types import (
    GrantFinderState, LogConfig, OutputConfig, 
    UserInputConfig, GrantSearchError
)
from .nodes import build_graph

# Initialize logger at module level
logger = logging.getLogger('grant_finder')

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

def print_progress(state: GrantFinderState):
    """Print current search progress"""
    progress = state.search_progress
    print(f"\nSearch Progress:")
    print(f"Sources Processed: {len(progress['sources_searched'])}/{progress['total_sources']}")
    print(f"Successful Searches: {progress['successful_searches']}")
    print(f"Failed Searches: {progress['failed_searches']}")
    if state.identified_gaps:
        print("\nIdentified Gaps:")
        for gap in state.identified_gaps:
            print(f"- {gap['type']}: {gap['description']}")

def save_results(
    state: GrantFinderState,
    output_dir: Path,
    timestamp: str
) -> None:
    """Save all results and generate summary"""
    results_dir = output_dir / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save company profile
    with open(results_dir / "01_company_profile.json", 'w') as f:
        json.dump(state["company_profile"], f, indent=2)
    
    # Save search requirements
    with open(results_dir / "02_search_requirements.json", 'w') as f:
        json.dump(state["search_requirements"], f, indent=2)
    
    # Save grant opportunities
    with open(results_dir / "03_grant_opportunities.json", 'w') as f:
        json.dump(state["grant_opportunities"], f, indent=2)
    
    # Save funding source tracking
    with open(results_dir / "04_funding_source_tracking.json", 'w') as f:
        json.dump(state["funding_sources"], f, indent=2)
    
    # Save final report
    with open(results_dir / "05_final_report.json", 'w') as f:
        json.dump(state["final_report"], f, indent=2)
    
    # Generate and save summary
    summary = {
        "timestamp": timestamp,
        "total_sources_processed": len(state["funding_sources"]),
        "total_opportunities_found": len(state["grant_opportunities"]),
        "successful_searches": sum(
            1 for s in state["funding_sources"].values() 
            if s["search_successful"]
        ),
        "errors": state["errors"]
    }
    
    with open(results_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print("\n" + "="*50)
    print("Grant Search Summary")
    print("="*50)
    print(f"Total Sources Processed: {summary['total_sources_processed']}")
    print(f"Successful Searches: {summary['successful_searches']}")
    print(f"Total Opportunities Found: {summary['total_opportunities_found']}")
    if summary['errors']:
        print("\nErrors encountered:")
        for error in summary['errors']:
            print(f"- {error}")
    print("\nResults saved to:", results_dir)
    print("="*50 + "\n")

def main():
    try:
        # Load environment variables
        load_dotenv()
        
        # Generate timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Load configuration
        config = load_config("config/user_config.yaml")
        llm_config = get_llm_config(config)
        output_dir = Path(config["output"]["output_directory"])
        
        # Setup logging
        log_config = setup_logging(timestamp, output_dir)
        logger = log_config["system_logger"]
        
        logger.info("Starting Grant Finder process")
        
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
                # Get API keys
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise GrantSearchError("OpenAI API key not found")
                
                # Get user input for new search
                company_focus = get_user_input(
                    "What is the company's main focus for pursuing SBIR/STTR grants?",
                    "Artificial Intelligence and Machine Learning"
                )
                
                organization_focus = get_user_input(
                    "What Grant organization are you interested in (DoD, EPA, Army, Air Force, etc.)?",
                    "DoD"
                )
        else:
            # Get API keys
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise GrantSearchError("OpenAI API key not found")
            
            # Get user input for new search
            company_focus = get_user_input(
                "What is the company's main focus for pursuing SBIR/STTR grants?",
                "Artificial Intelligence and Machine Learning"
            )
            
            organization_focus = get_user_input(
                "What Grant organization are you interested in (DoD, EPA, Army, Air Force, etc.)?",
                "DoD"
            )
        
        # Initialize LLM
        llm = ChatOpenAI(**llm_config, api_key=api_key)

        # Build workflow
        workflow = build_graph(
            company_context_path=config["company_context"]["directory"],
            funding_sources_path=config["funding_sources"]["file_path"],
            output_dir=output_dir,
            logger=logger,
            llm=llm,
            api_key=api_key
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
        
        # Compile and execute workflow
        logger.info("Executing grant search workflow")
        app = workflow.compile()
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