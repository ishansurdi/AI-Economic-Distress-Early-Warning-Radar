from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
import pandas as pd
from pathlib import Path

from backend.utils.logger import get_logger
from backend.config.settings import settings
from backend.services.llm_service import LLMService

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1", tags=["recommendations"])

# Initialize LLM service
llm_service = LLMService()


class RecommendationRequest(BaseModel):
    risk_score: float
    cashflow_summary: Dict
    anomalies: List[Dict]
    invoice_data: List[Dict]


@router.post("/recommend")
async def get_recommendations(request: RecommendationRequest):
    
    logger.info("Generating recommendations")
    
    try:
        recommendations = llm_service.generate_recommendations(
            risk_score=request.risk_score,
            cashflow_summary=request.cashflow_summary,
            anomalies=request.anomalies,
            invoice_data=request.invoice_data
        )
        
        return {
            "status": "success",
            "recommendations": recommendations,
            "count": len(recommendations)
        }
    
    except Exception as e:
        logger.error(f"Recommendation generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate recommendations: {str(e)}")


@router.get("/recommend/quick")
async def get_quick_recommendations():
    
    logger.info("Generating quick recommendations")
    
    try:
        # Generate generic recommendations
        recommendations = llm_service._generate_rule_based(
            risk_score=50,
            cashflow_summary={},
            anomalies=[],
            invoice_data=[]
        )
        
        return {
            "status": "success",
            "recommendations": recommendations,
            "type": "generic"
        }
    
    except Exception as e:
        logger.error(f"Quick recommendation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
