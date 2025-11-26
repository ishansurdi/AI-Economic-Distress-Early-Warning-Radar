from fastapi import APIRouter
from datetime import datetime
import psutil
import os
from pathlib import Path
from backend.utils.logger import get_logger
from backend.config.settings import settings

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1", tags=["health"])


@router.get("/health")
async def health_check():
    
    try:
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        # Get disk for current directory (Windows compatible)
        current_drive = Path.cwd().drive or 'C:'
        disk = psutil.disk_usage(current_drive)
        
        # Check upload directories
        upload_dir_exists = settings.UPLOAD_DIR.exists()
        processed_dir_exists = settings.PROCESSED_DIR.exists()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "system": {
                "cpu_usage_percent": cpu_percent,
                "memory_usage_percent": memory.percent,
                "memory_available_mb": round(memory.available / (1024 * 1024), 2),
                "disk_usage_percent": disk.percent,
                "disk_free_gb": round(disk.free / (1024 * 1024 * 1024), 2)
            },
            "application": {
                "upload_dir": str(settings.UPLOAD_DIR),
                "upload_dir_exists": upload_dir_exists,
                "processed_dir": str(settings.PROCESSED_DIR),
                "processed_dir_exists": processed_dir_exists,
                "forecast_days": settings.FORECAST_DAYS,
                "max_file_size_mb": settings.MAX_FILE_SIZE_MB
            }
        }
    
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }


@router.get("/")
async def root():
    return {
        "application": "E-DERA API",
        "version": "1.0.0",
        "description": "AI Economic Distress Early-Warning Radar",
        "endpoints": {
            "upload": "/api/v1/upload",
            "analyse": "/api/v1/analyse",
            "recommend": "/api/v1/recommend",
            "health": "/api/v1/health"
        }
    }
