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

# Configuration - Update BASE_URL to your Heroku URL
BASE_URL = "https://your-app-name.herokuapp.com"
os.makedirs("uploads", exist_ok=True)
os.makedirs("static/thumbnails", exist_ok=True)

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
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Initial DB Entry
    new_job = JobHistory(
        id=job_id,
        status="processing",
        thumbnail_path=f"{BASE_URL}/static/thumbnails/{job_id}.jpg"
    )
    db.add(new_job)
    db.commit()
    
    background_tasks.add_task(process_video_task, job_id, file_path)
    return {"job_id": job_id}

@app.get("/history")
async def get_history(db: Session = Depends(get_db)):
    # Returns the list of jobs for the History screen
    return db.query(JobHistory).order_by(JobHistory.created_at.desc()).limit(50).all()

@app.get("/status/{job_id}")
async def get_status(job_id: str, db: Session = Depends(get_db)):
    job = db.query(JobHistory).filter(JobHistory.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id": job.id,
        "status": job.status,
        "result": job.result,
        "confidence": job.confidence
    }