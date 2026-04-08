import cv2
import numpy as np
import sqlite3
import os
from tensorflow.keras.models import load_model

# Load model ONCE
MODEL = load_model('models/Meso4_DF.h5')

def process_video_task(job_id, file_path):
    conn = sqlite3.connect("jobs.db")
    cursor = conn.cursor()
    predictions = []
    
    try:
        # Check if the file is an image or video based on extension
        ext = os.path.splitext(file_path)[1].lower()
        is_image = ext in ['.jpg', '.jpeg', '.png', '.webp']

        if is_image:
            # --- IMAGE PROCESSING ---
            frame = cv2.imread(file_path)
            if frame is not None:
                # Save thumbnail
                thumb = cv2.resize(frame, (150, 150))
                cv2.imwrite(f"static/thumbnails/{job_id}.jpg", thumb)
                
                # AI Prediction
                resized = cv2.resize(frame, (256, 256))
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                normalized = rgb.astype('float32') / 255.0
                input_data = np.expand_dims(normalized, axis=0)
                pred = MODEL.predict(input_data, verbose=0)[0][0]
                predictions.append(pred)
        else:
            # --- VIDEO PROCESSING ---
            cap = cv2.VideoCapture(file_path)
            success, first_frame = cap.read()
            if success:
                thumb = cv2.resize(first_frame, (150, 150))
                cv2.imwrite(f"static/thumbnails/{job_id}.jpg", thumb)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                if cap.get(cv2.CAP_PROP_POS_FRAMES) % 15 == 0:
                    resized = cv2.resize(frame, (256, 256))
                    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                    normalized = rgb.astype('float32') / 255.0
                    input_data = np.expand_dims(normalized, axis=0)
                    pred = MODEL.predict(input_data, verbose=0)[0][0]
                    predictions.append(pred)
            cap.release()

        # --- UPDATE DATABASE ---
        if predictions:
            avg_score = float(np.mean(predictions))
            result = "REAL" if avg_score > 0.5 else "FAKE"
            conf = avg_score if avg_score > 0.5 else 1.0 - avg_score
            
            cursor.execute(
                "UPDATE jobs SET status = ?, result = ?, confidence = ? WHERE id = ?",
                ("completed", result, round(float(conf), 4), job_id)
            )
        else:
            raise Exception("No frames/images could be processed")

    except Exception as e:
        print(f"Error for job {job_id}: {e}")
        cursor.execute("UPDATE jobs SET status = 'error' WHERE id = ?", (job_id,))
    
    finally:
        conn.commit()
        conn.close()
        if os.path.exists(file_path):
            os.remove(file_path)