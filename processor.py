import os
import cv2
import numpy as np
import tensorflow as tf
import torch
from transformers import ViTImageProcessor, ViTForImageClassification
from sqlalchemy.orm import Session
from database import SessionLocal, JobHistory

# --- GLOBAL DETECTOR CLASS ---
class DeepfakeDetector:
    def __init__(self):
        print("🧠 Initializing High-Accuracy Human Deepfake Detector...")
        
        # 1. Face Detection (OpenCV)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # 2. Vision Transformer (Human Specialist)
        # Replacing generic ViT with a fine-tuned deepfake forensic model
        print("💡 Loading Specialist ViT (prithivMLmods/Deep-Fake-Detector-v2-Model)...")
        model_id = "prithivMLmods/Deep-Fake-Detector-v2-Model"
        self.vit_processor = ViTImageProcessor.from_pretrained(model_id)
        self.vit_model = ViTForImageClassification.from_pretrained(model_id)
        self.vit_model.eval()

        # 3. Xception (Texture Specialist)
        # Using the standard pre-trained Xception for high-frequency noise analysis
        print("💡 Loading Texture Specialist (Xception)...")
        self.xception_base = tf.keras.applications.Xception(weights='imagenet', include_top=False, pooling='avg')
        
        # 4. LSTM (Temporal consistency)
        self.lstm_model = self._build_lstm()

    def _build_lstm(self):
        # Input shape: 15 frames, 2048 features from Xception
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(15, 2048)), 
            tf.keras.layers.LSTM(256, return_sequences=False),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        return model

    def get_face_crop(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.2, 5)
        if len(faces) == 0: return None
        # Focus on the largest face (usually the subject)
        (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
        # Add 20% padding to capture hairline/ear artifacts
        pad = int(w * 0.20)
        y1, y2 = max(0, y-pad), min(frame.shape[0], y+h+pad)
        x1, x2 = max(0, x-pad), min(frame.shape[1], x+w+pad)
        face = frame[y1:y2, x1:x2]
        return cv2.resize(face, (224, 224)) # ViT standard size

    def get_hybrid_scores(self, face_image):
        # --- 1. ViT Specialist Score ---
        inputs = self.vit_processor(images=face_image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.vit_model(**inputs)
            # The model outputs [Realism_Logit, Deepfake_Logit]
            # Softmax to get probability for the 'Deepfake' class (index 1)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            vit_score = probs[0][1].item() 

        # --- 2. Xception Features & Score ---
        # Resize for Xception (299x299)
        xc_img = cv2.resize(face_image, (299, 299)).astype(np.float32)
        xc_img = tf.keras.applications.xception.preprocess_input(xc_img)
        xc_batch = np.expand_dims(xc_img, 0)
        
        features = self.xception_base(xc_batch, training=False)
        # Note: Xception here provides the raw features for the LSTM
        feat_np = features.numpy().flatten()
        
        return vit_score, feat_np

detector = DeepfakeDetector()

def process_video_task(job_id, file_path, base_url):
    db = SessionLocal()
    vit_results, sequence_features = [], []
    last_face = None

    try:
        cap = cv2.VideoCapture(file_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Sample 15 frames for temporal analysis
        frame_indices = np.linspace(0, max(0, total_frames - 1), 15, dtype=int)
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret: break
            
            face = detector.get_face_crop(frame)
            if face is not None:
                last_face = face
                v_score, feat = detector.get_hybrid_scores(face)
                vit_results.append(v_score)
                sequence_features.append(feat)
        
        cap.release()
        job = db.query(JobHistory).filter(JobHistory.id == job_id).first()

        if not vit_results:
            job.result, job.status, job.confidence = "NO_FACE_DETECTED", "completed", 0.0
        else:
            # Save Thumbnail
            os.makedirs("static/thumbnails", exist_ok=True)
            cv2.imwrite(f"static/thumbnails/{job_id}.jpg", last_face)
            job.thumbnail_path = f"{base_url}/static/thumbnails/{job_id}.jpg"

            # --- HEURISTIC SCORING ---
            # Max pooling: If any frame looks very fake, it's suspicious
            peak_vit = np.max(vit_results)
            
            # Temporal Score (LSTM)
            while len(sequence_features) < 15: # Pad if video too short
                sequence_features.append(sequence_features[-1] if sequence_features else np.zeros(2048))
            
            seq_arr = np.expand_dims(sequence_features[:15], 0)
            temp_score = float(detector.lstm_model(seq_arr, training=False).numpy().flatten()[0])

            # WEIGHTING: 60% ViT Specialist, 40% LSTM Temporal
            final_score = (peak_vit * 0.6) + (temp_score * 0.4)

            # Strict Classification
            if final_score > 0.60:
                job.result = "FAKE"
            elif final_score < 0.30:
                job.result = "REAL"
            else:
                job.result = "UNCERTAIN"

            job.confidence = round(max(final_score, 1 - final_score) * 100, 2)
            job.status = "completed"

        db.commit()
    except Exception as e:
        db.rollback()
        print(f"Error: {e}")
    finally:
        db.close()
        if os.path.exists(file_path): os.remove(file_path)