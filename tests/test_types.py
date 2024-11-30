# tests/test_types.py
import pytest
from grant_finder.types import (
    GrantOpportunityState,
    ValidationError
)

def test_grant_opportunity_state_validation(sample_grant_opportunity):
    try:
        opportunity = GrantOpportunityState(**sample_grant_opportunity)
        assert opportunity.topic_id == "AF241-001"
        assert opportunity.alignment_score == 85.5
    except ValidationError as e:
        pytest.fail(f"Validation failed: {str(e)}")

def test_grant_opportunity_invalid_data():
    invalid_data = {
        "topic_id": "invalid",  # Invalid format
        "title": "",  # Empty
        "description": "too short",  # Too short
        "technical_requirements": [],  # Empty list
        "alignment_score": "invalid",  # Wrong type
        "award_amount": "invalid",
        "deadline": "invalid",
        "url": "invalid"
    }
    
    with pytest.raises(ValidationError):
        GrantOpportunityState(**invalid_data)