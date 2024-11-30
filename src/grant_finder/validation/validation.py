# New code (in green):
from datetime import datetime
import re
from typing import Optional
import logging
from pydantic import BaseModel, Field, validator, root_validator
from grant_finder.types import GrantOpportunityState, ValidationError

logger = logging.getLogger('grant_finder')

class GrantValidationModel(BaseModel):
    topic_id: str = Field(..., description="Grant topic ID")
    title: str = Field(..., min_length=3, description="Grant title")
    description: str = Field(..., min_length=100, description="Detailed grant description")
    technical_requirements: list[str] = Field(..., min_items=1, description="Technical requirements")
    alignment_score: float = Field(..., ge=0, le=100, description="Alignment score (0-100)")
    award_amount: str = Field(..., description="Grant award amount")
    deadline: str = Field(..., description="Submission deadline")
    focus_areas: list[str] = Field(..., min_items=1, description="Focus areas")
    url: str = Field(..., description="Grant URL")
    contact_info: Optional[str] = Field("", description="Contact information")
    justification: Optional[str] = Field("", description="Alignment justification")

    @validator('topic_id')
    def validate_topic_id(cls, v):
        patterns = [
            r'^[A-Z]{2,4}\d{3}-\d{3}$',          # AF241-001
            r'^[A-Z]{2,4}\d{2}-[A-Z]-\d{3}$',    # AF24-T-001
            r'^[A-Z]{2,4}-\d{2}-\d{3}$',         # AFRL-24-001
            r'^\d{2}-\d{3}$'                      # 24-001
        ]
        if not any(re.match(pattern, v) for pattern in patterns):
            raise ValueError("Invalid topic ID format")
        return v

    @validator('deadline')
    def validate_deadline(cls, v):
        date_formats = ['%Y-%m-%d', '%m/%d/%Y', '%d-%m-%Y']
        
        for fmt in date_formats:
            try:
                deadline_date = datetime.strptime(v, fmt)
                if deadline_date > datetime.now():
                    return v
            except ValueError:
                continue
        
        raise ValueError("Invalid deadline format or past date")

    @validator('award_amount')
    def validate_award_amount(cls, v):
        # Remove currency symbols and commas
        cleaned = re.sub(r'[$,]', '', v)
        try:
            value = float(cleaned)
            if value <= 0:
                raise ValueError
            return v
        except ValueError:
            raise ValueError("Invalid award amount format")

    @validator('url')
    def validate_url(cls, v):
        url_pattern = re.compile(
            r'^https?://'
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'
            r'localhost|'
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'
            r'(?::\d+)?'
            r'(?:/?|[/?]\S+)$', re.IGNORECASE
        )
        if not url_pattern.match(v):
            raise ValueError("Invalid URL format")
        return v

    @root_validator
    def validate_complete_opportunity(cls, values):
        """Additional cross-field validations"""
        if values.get('alignment_score', 0) > 80:
            if not values.get('justification'):
                raise ValueError("Justification required for high alignment scores")
        return values

class GrantValidator:
    """Validates grant opportunities using Pydantic models"""
    
    @staticmethod
    def validate_opportunity(data: dict) -> GrantOpportunityState:
        """Validate and create a GrantOpportunityState instance"""
        try:
            # First validate using the detailed validation model
            validated_data = GrantValidationModel(**data)
            
            # Convert to GrantOpportunityState
            return GrantOpportunityState(**validated_data.dict())
            
        except Exception as e:
            logger.error(f"Validation error: {str(e)}")
            raise ValidationError(f"Grant opportunity validation failed: {str(e)}")

    @staticmethod
    def validate_opportunities(opportunities: list[dict]) -> list[GrantOpportunityState]:
        """Validate a list of grant opportunities"""
        validated_opportunities = []
        for opp in opportunities:
            try:
                validated_opp = GrantValidator.validate_opportunity(opp)
                validated_opportunities.append(validated_opp)
            except ValidationError as e:
                logger.warning(f"Skipping invalid opportunity: {str(e)}")
        return validated_opportunities