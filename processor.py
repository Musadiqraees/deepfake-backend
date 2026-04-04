import cv2
import numpy as np
import sqlite3
import os
from tensorflow.keras.models import load_model

def process_video_task(job_id, file_path):
    try:
        # 1. GENERATE THUMBNAIL (Immediate)
        cap = cv2.VideoCapture(file_path)
        success, first_frame = cap.read()
        if success:
            # Resize for mobile list speed (150x150 is perfect for HistoryCard)
            thumb = cv2.resize(first_frame, (150, 150))
            # Ensure the directory exists before writing
            os.makedirs("static/thumbnails", exist_ok=True)
            cv2.imwrite(f"static/thumbnails/{job_id}.jpg", thumb)
        
        # IMPORTANT: Reset video pointer to the start for AI processing
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # 2. RUN AI DETECTION (MesoNet)
        model = load_model('models/Meso4_DF.h5')
        predictions = []
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # Use your skip-frame optimization (every 15th frame)
            if frame_count % 15 == 0:
                resized = cv2.resize(frame, (256, 256))
                normalized = resized.astype('float32') / 255.0
                input_data = np.expand_dims(normalized, axis=0)
                pred = model.predict(input_data, verbose=0)[0][0]
                predictions.append(pred)
            frame_count += 1
        
        cap.release()

        # 3. CALCULATE FINAL RESULTS
        avg_score = np.mean(predictions)
        result = "REAL" if avg_score > 0.5 else "FAKE"
        confidence = float(avg_score if avg_score > 0.5 else 1.0 - avg_score)

        # 4. UPDATE DB
        conn = sqlite3.connect("jobs.db")
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE jobs SET status = ?, result = ?, confidence = ? WHERE id = ?",
            ("completed", result, round(confidence, 4), job_id)
        )
        conn.commit()
        conn.close()

    except Exception as e:
        print(f"Error: {e}")
        # Update DB to 'error' status so Android app stops spinning
        conn = sqlite3.connect("jobs.db")
        conn.execute("UPDATE jobs SET status = 'error' WHERE id = ?", (job_id,))
        conn.commit()
        conn.close()
    
    finally:
        # Clean up the original upload to save server space
        if os.path.exists(file_path):
            os.remove(file_path)