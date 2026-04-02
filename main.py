import os
import uuid
import sqlite3
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from processor import process_video_task

app = FastAPI()

# 1. ALLOW MOBILE ACCESS (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. INITIALIZE DATABASE
def init_db():
    conn = sqlite3.connect("jobs.db")
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS jobs 
                     (id TEXT PRIMARY KEY, status TEXT, result TEXT, confidence REAL)''')
    conn.commit()
    conn.close()

init_db()

@app.post("/detect")
async def detect_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())
    
    # Use your "uploads" folder
    os.makedirs("uploads", exist_ok=True)
    file_path = os.path.join("uploads", f"{job_id}_{file.filename}")
    
    # Save the uploaded file
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    
    # Insert "processing" status
    conn = sqlite3.connect("jobs.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO jobs (id, status) VALUES (?, ?)", (job_id, "processing"))
    conn.commit()
    conn.close()
    
    # Start the background AI task
    background_tasks.add_task(process_video_task, job_id, file_path)
    
    return {"job_id": job_id}

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    conn = sqlite3.connect("jobs.db")
    cursor = conn.cursor()
    cursor.execute("SELECT status, result, confidence FROM jobs WHERE id = ?", (job_id,))
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return {
        "job_id": job_id,
        "status": row[0],
        "result": row[1],
        "confidence": row[2]
    }