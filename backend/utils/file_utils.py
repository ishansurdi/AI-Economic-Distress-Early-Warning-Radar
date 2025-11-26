import os
import uuid
from pathlib import Path
from typing import Optional
from fastapi import UploadFile, HTTPException
from backend.config.settings import settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)

def validate_file_extension(filename: str, allowed_extensions: set = None) -> bool:
    if allowed_extensions is None:
        allowed_extensions = settings.ALLOWED_EXTENSIONS
    
    file_ext = Path(filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        logger.warning(f"Invalid file extension: {file_ext}")
        return False
    return True

async def save_upload_file(file: UploadFile, upload_dir: Path) -> Path:
    try:
        # Ensure directory exists
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate unique filename
        unique_filename = generate_unique_filename(file.filename)
        destination = upload_dir / unique_filename
        
        # Read file content
        content = await file.read()
        
        # Check file size
        if len(content) > settings.MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File size exceeds maximum allowed size of {settings.MAX_UPLOAD_SIZE / (1024*1024)}MB"
            )
        
        # Write file
        with open(destination, "wb") as f:
            f.write(content)
        
        logger.info(f"File saved successfully: {destination}")
        return destination
    
    except Exception as e:
        logger.error(f"Error saving file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

def generate_unique_filename(original_filename: str) -> str:
    ext = Path(original_filename).suffix
    unique_id = uuid.uuid4().hex[:8]
    return f"{Path(original_filename).stem}_{unique_id}{ext}"

def cleanup_temp_files(directory: Path, older_than_hours: int = 24):
    import time
    current_time = time.time()
    cutoff_time = current_time - (older_than_hours * 3600)
    
    try:
        for file_path in directory.iterdir():
            if file_path.is_file():
                file_modified_time = file_path.stat().st_mtime
                if file_modified_time < cutoff_time:
                    file_path.unlink()
                    logger.info(f"Deleted old file: {file_path}")
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")
