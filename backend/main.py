from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
from pathlib import Path

from backend.config.settings import settings
from backend.utils.logger import get_logger
from backend.routers import upload, analyse, recommend, health

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting E-DERA API")
    
    # Create necessary directories
    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    settings.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Upload directory: {settings.UPLOAD_DIR}")
    logger.info(f"Processed directory: {settings.PROCESSED_DIR}")
    logger.info(f"Model directory: {settings.MODELS_DIR}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down E-DERA API")


# Initialize FastAPI app
app = FastAPI(
    title="E-DERA API",
    description="AI Economic Distress Early-Warning Radar - Financial Analysis Platform",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router)
app.include_router(upload.router)
app.include_router(analyse.router)
app.include_router(recommend.router)

logger.info("All routers registered successfully")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "application": "E-DERA API",
        "version": "1.0.0",
        "description": "AI Economic Distress Early-Warning Radar",
        "status": "running",
        "docs": "/docs",
        "endpoints": {
            "health": "/api/v1/health",
            "upload": "/api/v1/upload",
            "analyse": "/api/v1/analyse",
            "recommend": "/api/v1/recommend"
        }
    }


if __name__ == "__main__":
    logger.info("Starting server on http://0.0.0.0:8000")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
