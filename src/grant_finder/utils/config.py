from pathlib import Path
from typing import Dict, Any
import yaml
from pydantic import ValidationError
import logging

from ..types import UserInputConfig, AgentConfig, ValidationError

logger = logging.getLogger('grant_finder')

def load_yaml_config(file_path: Path) -> Dict[str, Any]:
    """Load and validate YAML configuration file"""
    try:
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config from {file_path}: {str(e)}")
        raise ValidationError(f"Config loading failed: {str(e)}")

def validate_paths(config: Dict[str, Any]) -> None:
    """Validate all paths in configuration exist"""
    required_paths = [
        ("company_context.directory", "Company context directory"),
        ("funding_sources.file_path", "Funding sources file"),
        ("output.output_directory", "Output directory")
    ]
    
    # Add GPT-J specific path validation
    if config.get('embeddings', {}).get('provider') == 'gptj':
        model_path = config.get('embeddings', {}).get('gptj', {}).get('model_path')
        if not model_path or not Path(model_path).exists():
            raise ValidationError(f"GPT-J model not found at: {model_path}")
    
    for path_key, description in required_paths:
        current = config
        for key in path_key.split('.'):
            current = current.get(key, {})
        
        if not current or not Path(current).exists():
            raise ValidationError(f"{description} not found: {current}")

def load_agent_configs() -> Dict[str, AgentConfig]:
    """Load and validate agent configurations"""
    config_path = Path(__file__).parent.parent / 'config' / 'agents.yaml'
    config = load_yaml_config(config_path)
    
    try:
        return {
            name: AgentConfig(**cfg)
            for name, cfg in config['agents'].items()
        }
    except ValidationError as e:
        logger.error(f"Agent config validation failed: {str(e)}")
        raise

def load_user_config() -> UserInputConfig:
    """Load and validate user configuration"""
    config_path = Path(__file__).parent.parent / 'config' / 'user_config.yaml'
    config = load_yaml_config(config_path)
    
    try:
        # Validate paths before creating UserInputConfig 
        validate_paths(config)
        return UserInputConfig (**config)
    except ValidationError as e:
        logger.error(f"User config validation failed: {str(e)}")
        raise

def get_llm_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get LLM configuration from main config"""
    model_type = config.get('llm', {}).get('model_type', 'openai')
    
    if model_type == 'openai':
        return config.get('llm', {}).get('openai', {})
    elif model_type == 'bitnet':
        bitnet_config = config.get('llm', {}).get('bitnet', {})
        # Ensure required paths are present
        if 'cli_path' not in bitnet_config:
            raise ValueError("BitNet config missing required 'cli_path'")
        if 'model_path' not in bitnet_config:
            raise ValueError("BitNet config missing required 'model_path'")
        return bitnet_config
    else:
        raise ValueError(f"Unknown model type: {model_type}")