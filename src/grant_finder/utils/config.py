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
    
    for path_key, description in required_paths:
        # Navigate nested dictionary using path_key
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
    """Get LLM configuration based on the model type"""
    model_type = config.get("llm", {}).get("model_type", "openai")
    llm_config = config.get("llm", {})
    
    if model_type == "openai":
        openai_config = llm_config.get("openai", {})
        return {
            "model_name": openai_config.get("model_name", "gpt-3.5-turbo-0125"),
            "temperature": openai_config.get("temperature", 0.0),
            "request_timeout": openai_config.get("request_timeout", 180),
            "max_tokens": openai_config.get("max_tokens", 500),
            "max_retries": openai_config.get("max_retries", 3),
            "concurrency_limit": openai_config.get("concurrency_limit", 2),
            "backoff_factor": openai_config.get("backoff_factor", 2),
            "min_delay": openai_config.get("min_delay", 1)
        }
    elif model_type == "bitnet":
        bitnet_config = llm_config.get("bitnet", {})
        return {
            "model_path": bitnet_config.get("model_path", "models/bitnet_b1_58-large"),
            "model_size": bitnet_config.get("model_size", "0.7B"),
            "quantization": bitnet_config.get("quantization", "i2_s"),
            "kernel_type": bitnet_config.get("kernel_type", "i2_s"),
            "threads": bitnet_config.get("threads", 4),
            "ctx_size": bitnet_config.get("ctx_size", 2048),
            "temperature": bitnet_config.get("temperature", 0.0),
            "n_predict": bitnet_config.get("n_predict", 128),
            "n_prompt": bitnet_config.get("n_prompt", 512),
            "quant_embd": bitnet_config.get("quant_embd", False),
            "use_pretuned": bitnet_config.get("use_pretuned", False)
        }
    else:
        raise ValueError(f"Unsupported model type: {model_type}")