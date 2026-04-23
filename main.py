import os
import uuid
import shutil
import subprocess
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from database import get_db, SessionLocal, JobHistory
from processor import process_video_task

app = FastAPI()

# --- Configuration ---
BASE_URL = "https://1071-2003-fc-7f1f-b5df-b930-d7b-a375-7ee6.ngrok-free.app"
os.makedirs("uploads", exist_ok=True)
os.makedirs("static/thumbnails", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- Normalization Helper ---
def normalize_video(input_path, output_path):
    """Ensures videos are readable by AI. Static images should bypass this."""
    try:
        command = [
            'ffmpeg', '-i', input_path,
            '-vcodec', 'libx264', 
            '-preset', 'veryfast', 
            '-crf', '28',         
            '-acodec', 'aac', 
            output_path, '-y'
        ]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception as e:
        print(f"FFmpeg Error: {e}")
        return False

# --- Modified Detection Endpoint ---
@app.post("/detect")
async def detect_media(
    background_tasks: BackgroundTasks, 
    file: UploadFile = File(...), 
    db: Session = Depends(get_db)
):
    job_id = str(uuid.uuid4())
    _, ext = os.path.splitext(file.filename.lower() if file.filename else ".mp4")
    
    # Save original file
    raw_path = os.path.join("uploads", f"raw_{job_id}{ext}")
    # Final path: If image, keep original extension; if video, normalize to .mp4
    is_image = ext in ['.jpg', '.jpeg', '.png']
    final_ext = ext if is_image else ".mp4"
    final_path = os.path.join("uploads", f"{job_id}{final_ext}")
    
    with open(raw_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    new_job = JobHistory(
        id=job_id,
        status="processing",
        thumbnail_path=f"{BASE_URL}/static/thumbnails/{job_id}.jpg"
    )
    db.add(new_job)
    db.commit()
    
    # Pass whether it's an image to the task handler
    background_tasks.add_task(handle_whatsapp_task, job_id, raw_path, final_path, BASE_URL, is_image)
    
    return {"job_id": job_id}

def handle_whatsapp_task(job_id, raw_path, final_path, base_url, is_image):
    success = True
    
    if is_image:
        # 🟢 IMAGE PATH: Skip FFmpeg, just move raw to final
        try:
            shutil.move(raw_path, final_path)
        except Exception as e:
            print(f"File Move Error: {e}")
            success = False
    else:
        # 🔵 VIDEO PATH: Normalize using FFmpeg
        success = normalize_video(raw_path, final_path)
        if os.path.exists(raw_path):
            os.remove(raw_path)

    if success:
        # Proceed to AI analysis
        process_video_task(job_id, final_path, base_url)
    else:
        # Handle failure
        db = SessionLocal()
        job = db.query(JobHistory).filter(JobHistory.id == job_id).first()
        if job:
            job.status = "error"
            job.result = "Failed to process media format"
            db.commit()
        db.close()

# --- Standard History/Status Endpoints ---
@app.get("/history")
async def get_history(db: Session = Depends(get_db)):
    return db.query(JobHistory).order_by(JobHistory.created_at.desc()).limit(50).all()

@app.get("/status/{job_id}")
async def get_status(job_id: str, db: Session = Depends(get_db)):
    job = db.query(JobHistory).filter(JobHistory.id == job_id).first()
    if not job: raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id": job.id,
        "status": job.status,   
        "result": job.result,   
        "confidence": job.confidence,
        "thumbnail_path": job.thumbnail_path
    }