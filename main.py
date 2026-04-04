import os
import uuid
import sqlite3
import shutil
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from processor import process_video_task

app = FastAPI()

# Ensure directories exist
os.makedirs("uploads", exist_ok=True)
os.makedirs("static/thumbnails", exist_ok=True)

# MOUNT STATIC FOLDER: This lets Android load images via URL
# e.g., http://your-app.herokuapp.com/static/thumbnails/uuid.jpg
app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def init_db():
    conn = sqlite3.connect("jobs.db")
    cursor = conn.cursor()
    # Added thumbnail_path and created_at columns
    cursor.execute('''CREATE TABLE IF NOT EXISTS jobs 
                     (id TEXT PRIMARY KEY, 
                      status TEXT, 
                      result TEXT, 
                      confidence REAL,
                      thumbnail_path TEXT,
                      created_at TEXT)''')
    conn.commit()
    conn.close()

init_db()

@app.post("/detect")
async def detect_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())
    file_extension = file.filename.split(".")[-1]
    file_path = os.path.join("uploads", f"{job_id}.{file_extension}")
    
    # Save file safely
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Prepare metadata for Android History tab
    # Change 'your-app-name' to your actual Heroku URL
    base_url = "https://your-app-name.herokuapp.com" 
    thumb_url = f"{base_url}/static/thumbnails/{job_id}.jpg"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Insert initial record
    conn = sqlite3.connect("jobs.db")
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO jobs (id, status, thumbnail_path, created_at) VALUES (?, ?, ?, ?)", 
        (job_id, "processing", thumb_url, timestamp)
    )
    conn.commit()
    conn.close()
    
    # Pass to background worker
    background_tasks.add_task(process_video_task, job_id, file_path)
    
    return {"job_id": job_id}

@app.get("/history")
async def get_history():
    conn = sqlite3.connect("jobs.db")
    conn.row_factory = sqlite3.Row # Returns rows as dictionaries
    cursor = conn.cursor()
    # Fetch latest 50 scans
    cursor.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT 50")
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]

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