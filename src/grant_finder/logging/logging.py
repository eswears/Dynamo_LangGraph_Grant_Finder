from pathlib import Path
import logging
from rich.logging import RichHandler
from rich.console import Console
from rich.traceback import install
from datetime import datetime
from typing import Tuple

# Install rich traceback handler
install(show_locals=True)

def create_log_formatters() -> Tuple[logging.Formatter, logging.Formatter]:
    """Create formatters for file and console logging"""
    file_formatter = logging.Formatter(
        '[%(asctime)s] %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_formatter = logging.Formatter(
        '%(message)s'
    )
    
    return file_formatter, console_formatter

def setup_logger(
    name: str,
    log_dir: Path,
    level: int = logging.INFO,
    console: bool = True
) -> logging.Logger:
    """Setup and configure a logger"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create formatters
    file_formatter, console_formatter = create_log_formatters()
    
    # File handler
    file_handler = logging.FileHandler(
        log_dir / f"{name.replace('.', '_')}.log",
        encoding='utf-8'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler with rich formatting
    if console:
        console_handler = RichHandler(
            console=Console(force_terminal=True),
            show_time=False,
            show_path=False
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    return logger
def setup_logging(
    output_dir: Path,
    timestamp: str
) -> Tuple[logging.Logger, logging.Logger]:
    """Setup all logging for the application"""
    # Create log directory
    log_dir = output_dir / timestamp / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup system logger
    system_logger = setup_logger(
        "grant_finder",
        log_dir,
        console=True
    )
    
    # Setup agent logger
    agent_logger = setup_logger(
        "grant_finder.agents",
        log_dir,
        console=False
    )
    
    return system_logger, agent_logger

class LoggerContextManager:
    """Context manager for temporary logging level changes"""
    def __init__(self, logger: logging.Logger, level: int):
        self.logger = logger
        self.level = level
        self.previous_level = logger.level
    
    def __enter__(self):
        self.logger.setLevel(self.level)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.setLevel(self.previous_level)
