import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
from database import SessionLocal, JobHistory

# Load model globally to save RAM (Use tensorflow-cpu in requirements.txt)
MODEL = load_model('models/Meso4_DF.h5')

def process_video_task(job_id, file_path):
    db = SessionLocal()
    predictions = []
    
    try:
        ext = os.path.splitext(file_path)[1].lower()
        is_image = ext in ['.jpg', '.jpeg', '.png', '.webp']

        if is_image:
            # --- IMAGE LOGIC ---
            frame = cv2.imread(file_path)
            if frame is not None:
                thumb = cv2.resize(frame, (150, 150))
                cv2.imwrite(f"static/thumbnails/{job_id}.jpg", thumb)
                
                resized = cv2.resize(frame, (256, 256))
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                normalized = rgb.astype('float32') / 255.0
                input_data = np.expand_dims(normalized, axis=0)
                pred = MODEL.predict(input_data, verbose=0)[0][0]
                predictions.append(pred)
        else:
            # --- VIDEO LOGIC ---
            cap = cv2.VideoCapture(file_path)
            count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                if count == 0: # Save first frame as thumbnail
                    thumb = cv2.resize(frame, (150, 150))
                    cv2.imwrite(f"static/thumbnails/{job_id}.jpg", thumb)
                
                if count % 15 == 0: # Sample frames
                    resized = cv2.resize(frame, (256, 256))
                    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                    normalized = rgb.astype('float32') / 255.0
                    input_data = np.expand_dims(normalized, axis=0)
                    pred = MODEL.predict(input_data, verbose=0)[0][0]
                    predictions.append(pred)
                count += 1
            cap.release()

        # Update Database Record
        job = db.query(JobHistory).filter(JobHistory.id == job_id).first()
        if job and predictions:
            avg_score = float(np.mean(predictions))
            # MesoNet: High score = Real
            job.result = "REAL" if avg_score > 0.5 else "FAKE"
            job.confidence = float(avg_score if avg_score > 0.5 else 1.0 - avg_score)
            job.status = "completed"
        else:
            job.status = "error"
            job.result = "Failed to process"

        db.commit()

    except Exception as e:
        print(f"Background Error: {e}")
        job = db.query(JobHistory).filter(JobHistory.id == job_id).first()
        if job:
            job.status = "error"
            db.commit()
    finally:
        db.close()
        if os.path.exists(file_path):
            os.remove(file_path)