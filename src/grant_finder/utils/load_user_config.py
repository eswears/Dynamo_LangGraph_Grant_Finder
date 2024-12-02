import os
import yaml
import logging
from pathlib import Path

logger = logging.getLogger('grant_finder')

def load_user_config() -> dict:
    """Load user configuration from YAML file"""
    try:
        # Get the config file path
        config_path = Path(__file__).parent.parent / 'config' / 'user_config.yaml'
        logger.info(('Config', 'PATH', f'Config path: {config_path}'))
        
        # Load the YAML file
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Validate required paths exist
        required_paths = {
            'Company Context': config['company_context']['directory'],
            'Output Directory': config['output']['output_directory'],
            'Funding Sources': config['funding_sources']['file_path']
        }
        
        for name, path in required_paths.items():
            if not os.path.exists(path):
                error_msg = f'{name} path not found: {path}'
                logger.error(('Config', 'ERROR', error_msg))
                raise FileNotFoundError(error_msg)
                
        # Special validation for funding sources file
        if not validate_funding_sources(config['funding_sources']['file_path']):
            raise ValueError('Invalid funding sources configuration')
            
        return config
        
    except Exception as e:
        logger.error(('Config', 'ERROR', f'Failed to load configuration: {str(e)}'))
        raise 