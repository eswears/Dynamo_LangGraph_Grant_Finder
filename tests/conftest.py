 tests/conftest.py
import pytest
from pathlib import Path
import tempfile
import yaml
import json
from unittest.mock import Mock
from langchain_openai import ChatOpenAI

@pytest.fixture
def mock_llm():
    return Mock(spec=ChatOpenAI)

@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture
def sample_config(temp_dir):
    config = {
        "company_context": {
            "directory": str(temp_dir / "company_docs")
        },
        "funding_sources": {
            "file_path": str(temp_dir / "sources.csv")
        },
        "output": {
            "format": "markdown",
            "save_to_file": True,
            "output_directory": str(temp_dir / "output")
        },
        "logging": {
            "level": "INFO",
            "file": str(temp_dir / "logs/grant_finder.log")
        }
    }
    
    # Create directories
    (temp_dir / "company_docs").mkdir()
    (temp_dir / "output").mkdir()
    
    # Create sample funding sources file
    with open(temp_dir / "sources.csv", 'w') as f:
        f.write("Title,Source\nDoD SBIR,https://www.dodsbir.net\n")
    
    # Create sample company document
    with open(temp_dir / "company_docs/profile.txt", 'w') as f:
        f.write("Company specializes in AI and ML solutions for defense applications.")
    
    return config

@pytest.fixture
def sample_grant_opportunity():
    return {
        "topic_id": "AF241-001",
        "title": "AI for Defense",
        "description": "Development of AI solutions for defense applications with focus on machine learning algorithms for threat detection.",
        "technical_requirements": ["ML expertise", "Defense domain knowledge"],
        "alignment_score": 85.5,
        "award_amount": "$1,000,000",
        "deadline": "2024-12-31",
        "focus_areas": ["AI", "Defense"],
        "url": "https://www.grants.gov/opportunity/AF241-001",
        "contact_info": "john.doe@af.mil",
        "justification": "Strong alignment with company expertise"
    }