import torch
import numpy as np
import cv2
from PIL import Image, ImageChops
from safetensors.torch import load_file
from timm import create_model

class DeepfakeDetector:
    def __init__(self, model_path='weights/model.safetensors'):
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        
        # --- LAYER 1: GENERATIVE (EfficientNet-B4 / ViT) ---
        # This uses your uploaded Safetensors to find texture glitches
        self.vit_model = create_model('efficientvit_b0', pretrained=False, num_classes=2)
        state_dict = load_file(model_path, device=str(self.device))
        self.vit_model.load_state_dict(state_dict)
        self.vit_model.to(self.device).eval()

    # --- LAYER 2: FORENSIC (ELA - Error Level Analysis) ---
    def get_ela_score(self, face_img):
        """Detects if face pixels have different compression than the background."""
        # Convert CV2 image to PIL
        pil_img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        
        # Save at lower quality then compare
        temp_file = "temp_ela.jpg"
        pil_img.save(temp_file, 'JPEG', quality=90)
        compressed_img = Image.open(temp_file)
        
        ela_diff = ImageChops.difference(pil_img, compressed_img)
        extrema = ela_diff.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        scale = 255.0 / (max_diff if max_diff > 0 else 1)
        
        # Return mean brightness of the difference (higher = more manipulated)
        return np.array(ela_diff).mean()

    # --- LAYER 3: BIOLOGICAL (rPPG - Heartbeat Consistency) ---
    def get_biological_score(self, frames):
        """Analyzes minute skin color changes (blood flow). AI faces don't 'pulse'."""
        green_channel_avg = []
        for frame in frames:
            # Focus on forehead/cheeks where blood flow is most visible
            roi = frame[20:50, 80:140] 
            green_channel_avg.append(np.mean(roi[:, :, 1]))
        
        # Calculate Variance: Real humans have a rhythmic pulse; Fakes are static or noisy
        variance = np.var(green_channel_avg)
        return 1.0 if variance < 0.01 else 0.0 # Low variance = High Fake probability

    # --- LAYER 4: PHYSICAL (Eye Reflection/Symmetry) ---
    def get_physical_score(self, face_img):
        """Checks if both eyes reflect light the same way."""
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        # Simplified: Check symmetry of brightest pixels (specular highlights)
        left_eye = gray[80:110, 60:90]
        right_eye = gray[80:110, 130:160]
        
        # Compare reflection patterns
        correlation = cv2.matchTemplate(left_eye, right_eye, cv2.TM_CCORR_NORMED)[0][0]
        return 1.0 - correlation # Lower correlation = More likely fake

    def analyze_full_forensics(self, frames, face_crop):
        # 1. Generative Score
        gen_score, _ = self.get_hybrid_scores(face_crop)
        
        # 2. Forensic Score
        forensic_score = self.get_ela_score(face_crop) / 10.0 # Normalized
        
        # 3. Biological Score (Requires multiple frames)
        bio_score = self.get_biological_score(frames)
        
        # 4. Physical Score
        phys_score = self.get_physical_score(face_crop)

        # FINAL WEIGHTED VOTE
        # Generative (40%) + Forensic (20%) + Biological (20%) + Physical (20%)
        final_score = (gen_score * 0.4) + (forensic_score * 0.2) + (bio_score * 0.2) + (phys_score * 0.2)
        
        return {
            "total_fake_probability": final_score,
            "breakdown": {
                "texture_glitch": gen_score,
                "compression_anomaly": forensic_score,
                "heartbeat_static": bio_score,
                "eye_asymmetry": phys_score
            }
        }