import os
import cv2
import numpy as np
import tensorflow as tf
import librosa
import soundfile as sf
from sqlalchemy.orm import Session
from database import SessionLocal, JobHistory
import subprocess

class DeepfakeDetector:
    def __init__(self):
        # --- 1. Model Setup ---
        self.MODEL_PATH = 'models/deepfake_detector_v2.h5'
        self.model = self._load_or_build_model()

        # --- 2. OpenCV Face Detection (The 3.13 Fix) ---
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # --- 3. Audio Model Setup ---
        self.AUDIO_MODEL_PATH = 'models/audio_deepfake_detector.h5'
        self.audio_model = self._load_or_build_audio_model()

        # --- 4. XceptionNet Model Setup ---
        self.XCEPTION_MODEL_PATH = 'models/xception_deepfake_detector.h5'
        self.xception_model = self._load_or_build_xception_model()

    def _load_or_build_model(self):
        os.makedirs("models", exist_ok=True)
        if os.path.exists(self.MODEL_PATH):
            print(f"✅ Loading existing EfficientNet model from {self.MODEL_PATH}...")
            return tf.keras.models.load_model(self.MODEL_PATH)
        
        print("🚀 Building EfficientNet-B0... (First run may take a minute)")
        base_model = tf.keras.applications.EfficientNetB0(
            weights='imagenet', include_top=False, input_shape=(224, 224, 3)
        )
        base_model.trainable = False 
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.save(self.MODEL_PATH)
        return model

    def _load_or_build_audio_model(self):
        os.makedirs("models", exist_ok=True)
        if os.path.exists(self.AUDIO_MODEL_PATH):
            print(f"✅ Loading existing audio model from {self.AUDIO_MODEL_PATH}...")
            return tf.keras.models.load_model(self.AUDIO_MODEL_PATH)

        print("🚀 Building CNN Audio Model...")
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(128, 431, 1)), # Example shape, will adjust based on actual Mel-spectrograms
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.save(self.AUDIO_MODEL_PATH)
        return model

    def _load_or_build_xception_model(self):
        os.makedirs("models", exist_ok=True)
        if os.path.exists(self.XCEPTION_MODEL_PATH):
            print(f"✅ Loading existing XceptionNet model from {self.XCEPTION_MODEL_PATH}...")
            return tf.keras.models.load_model(self.XCEPTION_MODEL_PATH)
        
        print("🚀 Building XceptionNet... (First run may take a minute)")
        base_model = tf.keras.applications.Xception(
            weights='imagenet', include_top=False, input_shape=(224, 224, 3)
        )
        base_model.trainable = False 
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.save(self.XCEPTION_MODEL_PATH)
        return model

    def get_face_crop(self, frame):
        """Detects and crops face using OpenCV Haar Cascades."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            return None
        
        # Pick the largest face found
        (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
        
        # Add 10% padding
        pad_w, pad_h = int(w * 0.1), int(h * 0.1)
        face = frame[max(0, y-pad_h):min(frame.shape[0], y+h+pad_h), 
                     max(0, x-pad_w):min(frame.shape[1], x+w+pad_h)]
        
        return cv2.resize(face, (224, 224))

    def predict_deepfake(self, face_image):
        """Makes a deepfake prediction on a pre-processed face image using EfficientNet."""
        img = tf.keras.applications.efficientnet.preprocess_input(face_image)
        return self.model.predict(np.expand_dims(img, 0), verbose=0)[0][0]

    def predict_xception_deepfake(self, face_image):
        """Makes a deepfake prediction on a pre-processed face image using XceptionNet."""
        img = tf.keras.applications.xception.preprocess_input(face_image)
        return self.xception_model.predict(np.expand_dims(img, 0), verbose=0)[0][0]


def extract_audio_from_video(video_path: str, audio_path: str) -> bool:
    """
    Extracts audio from a video file using ffmpeg.
    Returns True if successful, False otherwise.
    """
    command = [
        'ffmpeg',
        '-i', video_path,
        '-vn',  # No video
        '-acodec', 'pcm_s16le',  # Audio codec
        '-ar', '44100',  # Audio sample rate
        '-ac', '1',  # Mono audio
        audio_path
    ]
    try:
        # Run ffmpeg command, capture output for debugging
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        print(f"FFmpeg stdout: {result.stdout}")
        print(f"FFmpeg stderr: {result.stderr}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error extracting audio: {e}")
        print(f"FFmpeg stdout: {e.stdout}")
        print(f"FFmpeg stderr: {e.stderr}")
        return False
    except FileNotFoundError:
        print("Error: ffmpeg not found. Please install ffmpeg.")
        return False

def process_video_task(job_id, file_path, base_url): # Added base_url parameter
    db = SessionLocal()
    predictions = []
    last_face = None
    
    detector = DeepfakeDetector() # Instantiate the detector

    try:
        cap = cv2.VideoCapture(file_path)
        ext = os.path.splitext(file_path)[1].lower()
        is_image = ext in ['.jpg', '.jpeg', '.png', '.webp']

        audio_predictions = []
        audio_file_path = None

        if is_image:
            ret, frame = cap.read()
            if ret:
                face = detector.get_face_crop(frame)
                if face is not None:
                    last_face = face
                    efficientnet_pred = detector.predict_deepfake(face)
                    xception_pred = detector.predict_xception_deepfake(face)
                    print(f"EfficientNet Prediction: {efficientnet_pred}, XceptionNet Prediction: {xception_pred}")
                    # Simple average ensemble for now
                    predictions.append((efficientnet_pred + xception_pred) / 2.0)
        else:
            # Process video frames
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames > 0:
                frame_indices = np.linspace(0, total_frames - 1, 15, dtype=int)
                for idx in frame_indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if not ret: break
                    face = detector.get_face_crop(frame)
                    if face is not None:
                        last_face = face
                        efficientnet_pred = detector.predict_deepfake(face)
                        xception_pred = detector.predict_xception_deepfake(face)
                        print(f"EfficientNet Prediction: {efficientnet_pred}, XceptionNet Prediction: {xception_pred}")
                        # Simple average ensemble for now
                        predictions.append((efficientnet_pred + xception_pred) / 2.0)
            
            # Extract and process audio
            audio_file_path = f"/tmp/{job_id}.wav" # Temporary audio file
            if extract_audio_from_video(file_path, audio_file_path):
                audio_score = detector.predict_audio_deepfake(audio_file_path)
                print(f"Audio Model Prediction: {audio_score}")
                audio_predictions.append(audio_score)

        cap.release()

        # Early exit if no face was detected
        if last_face is None:
            job = db.query(JobHistory).filter(JobHistory.id == job_id).first()
            if job:
                job.status = "completed"
                job.result = "NO_FACE"
                job.confidence = 0.0
                job.thumbnail_path = f"{base_url}/static/thumbnails/no_face.jpg"
                db.commit()
            return

        # --- DATABASE UPDATE (Now inside the Try block) ---
        job = db.query(JobHistory).filter(JobHistory.id == job_id).first()
        if job:
            final_score = 0.0
            all_predictions = []
            if len(predictions) > 0:
                all_predictions.extend(predictions)
            if len(audio_predictions) > 0:
                all_predictions.extend(audio_predictions)
            
            if len(all_predictions) > 0:
                final_score = float(np.max(all_predictions))
            
            print(f"Final Score (Max Pooling): {final_score}")
            
            if len(all_predictions) > 0:
                # Save the thumbnail
                if last_face is not None:
                    thumb_path = f"static/thumbnails/{job_id}.jpg"
                    cv2.imwrite(thumb_path, last_face)
                    job.thumbnail_path = f"{base_url}/{thumb_path}"
                else:
                    # Use a placeholder thumbnail if no face was detected
                    job.thumbnail_path = f"{base_url}/static/thumbnails/no_face.jpg"

                # Classification will be handled after getting the "Uncertain" range
                # For now, keep the old classification logic, it will be updated in the next step
                if final_score > 0.60:
                    job.result = "FAKE"
                elif final_score < 0.35:
                    job.result = "REAL"
                else:
                    job.result = "UNCERTAIN"
                
                conf = final_score if final_score > 0.5 else 1.0 - final_score
                job.confidence = round(conf * 100, 2)
                job.status = "completed"
            else:
                # No predictions (e.g., no face detected, no audio extracted/processed)
                job.status = "completed"
                job.result = "NO_FACE"
                job.confidence = 0.0
            db.commit()

    except Exception as e:
        print(f"❌ Error: {e}")
        db.rollback()
        job = db.query(JobHistory).filter(JobHistory.id == job_id).first()
        if job:
            job.status = "error"
            job.result = str(e)
            db.commit()
    finally:
        db.close()
        if os.path.exists(file_path):
            os.remove(file_path)
        if audio_file_path and os.path.exists(audio_file_path):
            os.remove(audio_file_path)