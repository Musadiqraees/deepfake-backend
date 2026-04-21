import os
import uuid
import shutil
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from database import get_db, JobHistory
from processor import process_video_task

app = FastAPI()

# Configuration
BASE_URL = "https://0ae1-2003-fc-7f1f-b571-7948-fdb-3e21-79ff.ngrok-free.app"
os.makedirs("uploads", exist_ok=True)
os.makedirs("static/thumbnails", exist_ok=True)

# Mount static files so Android can download thumbnails
app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/detect")
async def detect_media(
    background_tasks: BackgroundTasks, 
    file: UploadFile = File(...), 
    db: Session = Depends(get_db)
):
    job_id = str(uuid.uuid4())
    _, ext = os.path.splitext(file.filename or ".jpg")
    file_path = os.path.join("uploads", f"{job_id}{ext}")
    
    # Save the uploaded file locally
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Initial DB Entry - Status is 'processing' until the AI finishes
    new_job = JobHistory(
        id=job_id,
        status="processing",
        # The thumbnail will be generated and saved at this path by the processor
        thumbnail_path=f"{BASE_URL}/static/thumbnails/{job_id}.jpg"
    )
    db.add(new_job)
    db.commit()
    
    # Fire and forget: AI starts in the background
    background_tasks.add_task(process_video_task, job_id, file_path)
    
    return {"job_id": job_id}

@app.get("/history")
async def get_history(db: Session = Depends(get_db)):
    # Returns the list of jobs for the Android History screen
    # Converts SQLAlchemy objects to a list of dicts automatically
    history = db.query(JobHistory).order_by(JobHistory.created_at.desc()).limit(50).all()
    return history

@app.get("/status/{job_id}")
async def get_status(job_id: str, db: Session = Depends(get_db)):
    job = db.query(JobHistory).filter(JobHistory.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # We return both 'status' and 'state' to ensure compatibility 
    # with different versions of your ServerApi.kt
    return {
        "job_id": job.id,
        "status": job.status,   
        "state": job.status,    
        "result": job.result,   # "FAKE" or "REAL"
        "confidence": job.confidence,
        "thumbnail_path": job.thumbnail_path
    }