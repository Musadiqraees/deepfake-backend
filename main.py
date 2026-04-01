from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
import uuid
import os
import shutil
from processor import process_video_task

app = FastAPI(title="Deepfake Detection Server")

# Simple in-memory dictionary. 
# NOTE: If you restart the server, this data clears (like an app's RAM).
jobs = {} 

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
async def root():
    return {"message": "Server is running locally"}

@app.post("/detect")
async def start_detection(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    # 1. Create a unique ID for this upload
    job_id = str(uuid.uuid4())
    file_extension = os.path.splitext(file.filename)[1]
    file_location = os.path.join(UPLOAD_DIR, f"{job_id}{file_extension}")
    
    # 2. Save the file to your Mac's disk
    try:
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save file: {e}")
        
    # 3. Initialize the job state
    jobs[job_id] = {"status": "processing", "result": None, "confidence": 0.0}
    
    # 4. Push to background (This returns immediately to the iOS app)
    background_tasks.add_task(process_video_task, job_id, file_location, jobs)
    
    return {"job_id": job_id, "status": "queued"}

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    result = jobs.get(job_id)
    if not result:
        raise HTTPException(status_code=404, detail="Job ID not found")
    return result

