import cv2
import numpy as np
import sqlite3
import os
import tensorflow as tf
from tensorflow.keras.models import load_model

# --- CONFIGURATION ---
MODEL_PATH = 'models/Meso4_DF.h5'

# Load model once at startup to save memory on the remote server
# We use a global variable so the model stays in RAM
model = None

def get_model():
    global model
    if model is None:
        print("Loading Meso4 Model into memory...")
        model = load_model(MODEL_PATH)
    return model

def update_db(job_id, status, result=None, confidence=None):
    """Writes the AI result back to the SQLite database"""
    conn = sqlite3.connect("jobs.db")
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE jobs SET status=?, result=?, confidence=? WHERE id=?", 
        (status, result, confidence, job_id)
    )
    conn.commit()
    conn.close()

def process_video_task(job_id: str, file_path: str):
    """The background task that runs the AI"""
    try:
        # Ensure model is loaded
        current_model = get_model()
        
        cap = cv2.VideoCapture(file_path)
        predictions = []
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Optimization: Only check every 15th frame (approx 2 frames per second)
            # This makes the remote server 15x faster without losing accuracy
            if frame_count % 15 == 0:
                # MesoNet expects 256x256
                resized = cv2.resize(frame, (256, 256))
                normalized = resized.astype('float32') / 255.0
                input_data = np.expand_dims(normalized, axis=0)
                
                # Predict (1.0 = Real, 0.0 = Fake)
                pred = current_model.predict(input_data, verbose=0)[0][0]
                predictions.append(pred)
            
            frame_count += 1
        
        cap.release()

        if not predictions:
            raise Exception("Could not extract frames from video.")

        # Calculate Final Results
        avg_score = np.mean(predictions)
        
        # Logic: MesoNet output > 0.5 is usually "Real"
        final_result = "REAL" if avg_score > 0.5 else "FAKE"
        
        # Calculate confidence percentage
        final_confidence = float(avg_score if avg_score > 0.5 else 1.0 - avg_score)

        # Update the DB so the /status/ endpoint can see it
        update_db(job_id, "completed", final_result, round(final_confidence, 4))
        print(f"Job {job_id} completed: {final_result}")

    except Exception as e:
        print(f"Error in background task: {e}")
        update_db(job_id, "error")
    
    finally:
        # CRITICAL: Delete the video file to prevent the server disk from filling up
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Temporary file {file_path} deleted.")