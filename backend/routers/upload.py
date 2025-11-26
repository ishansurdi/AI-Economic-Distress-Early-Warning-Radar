from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from typing import Optional
import os
from pathlib import Path
from backend.utils.logger import get_logger
from backend.utils.file_utils import validate_file_extension, save_upload_file
from backend.services.ingest_service import DataIngestService
from backend.config.settings import settings

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1", tags=["upload"])

# Ensure upload directories exist
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

@router.post("/upload")
async def upload_files(
    sales_file: Optional[UploadFile] = File(None),
    expense_file: Optional[UploadFile] = File(None)
):
    
    logger.info("Received file upload request")
    
    if not sales_file and not expense_file:
        raise HTTPException(status_code=400, detail="At least one file (sales or expense) is required")
    
    uploaded_files = {}
    
    try:
        # Process sales file
        if sales_file:
            if not validate_file_extension(sales_file.filename):
                raise HTTPException(status_code=400, detail=f"Invalid sales file type: {sales_file.filename}")
            
            sales_path = await save_upload_file(sales_file, settings.UPLOAD_DIR)
            uploaded_files['sales'] = str(sales_path)
            logger.info(f"Sales file saved: {sales_path}")
            
            # Validate sales data
            sales_df = DataIngestService.load_csv(sales_path)
            sales_result = DataIngestService.validate_sales_data(sales_df)
            sales_df = sales_result['dataframe']
            
            # Save processed data
            processed_sales_path = settings.PROCESSED_DIR / f"processed_{sales_path.name}"
            sales_df.to_csv(processed_sales_path, index=False)
            uploaded_files['sales_processed'] = str(processed_sales_path)
        
        # Process expense file
        if expense_file:
            if not validate_file_extension(expense_file.filename):
                raise HTTPException(status_code=400, detail=f"Invalid expense file type: {expense_file.filename}")
            
            expense_path = await save_upload_file(expense_file, settings.UPLOAD_DIR)
            uploaded_files['expense'] = str(expense_path)
            logger.info(f"Expense file saved: {expense_path}")
            
            # Validate expense data
            expense_df = DataIngestService.load_csv(expense_path)
            expense_result = DataIngestService.validate_expense_data(expense_df)
            expense_df = expense_result['dataframe']
            
            # Save processed data
            processed_expense_path = settings.PROCESSED_DIR / f"processed_{expense_path.name}"
            expense_df.to_csv(processed_expense_path, index=False)
            uploaded_files['expense_processed'] = str(processed_expense_path)
        
        logger.info("Files uploaded and validated successfully")
        
        return {
            "status": "success",
            "message": "Files uploaded and validated successfully",
            "files": uploaded_files,
            "sales_records": len(sales_df) if sales_file else 0,
            "expense_records": len(expense_df) if expense_file else 0
        }
    
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")


@router.get("/upload/history")
async def get_upload_history():
    
    try:
        upload_files = list(settings.UPLOAD_DIR.glob("*.csv"))
        
        history = [
            {
                "filename": f.name,
                "size_kb": round(f.stat().st_size / 1024, 2),
                "uploaded_at": f.stat().st_mtime
            }
            for f in upload_files
        ]
        
        # Sort by upload time, newest first
        history.sort(key=lambda x: x['uploaded_at'], reverse=True)
        
        return {
            "status": "success",
            "count": len(history),
            "files": history[:10]  # Last 10 uploads
        }
    
    except Exception as e:
        logger.error(f"Error fetching upload history: {e}")
        raise HTTPException(status_code=500, detail=str(e))
