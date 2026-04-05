from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import tensorflow as tf
import base64
from quality_check import check_image_quality

# --- 1. SETUP ---
app = FastAPI(title="Dental AI Stateless API")

# Enable CORS for Frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = 'dental_view_model_v2.keras'
CLASSES = [
    'Lower Front View', 'Lower Left View', 'Lower Occlusal View', 
    'Lower Right View', 'noise_objects', 'Upper Front View', 
    'Upper Left View', 'Upper Occlusal View', 'Upper Right View'
]

print("⏳ Loading AI Model into memory...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ AI Model Ready (Stateless Mode)")

# --- 2. ENHANCEMENT LOGIC ---
def apply_enhancements(image):
    """Processes the image in memory without saving to disk."""
    img_resized = cv2.resize(image, (1024, 1024))
    lab = cv2.cvtColor(img_resized, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

# --- 3. THE MAGIC ENDPOINT ---
@app.post("/analyze-view/")
async def analyze_view(
    file: UploadFile = File(...), 
    expected_view: str = Form(...) 
):
    try:
        # A. Decode image from network stream
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return {"match": "No", "processed_image": None}
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # B. Quality Check (Directly from RAM)
        is_valid, _, _ = check_image_quality(image_rgb)
        if not is_valid:
            return {"match": "No", "processed_image": None}

        # C. AI Prediction
        img_for_ai = cv2.resize(image_rgb, (224, 224))
        img_array = np.expand_dims(img_for_ai, axis=0) / 255.0
        preds = model.predict(img_array)
        predicted_view = CLASSES[np.argmax(preds)]

        # D. Match Logic (Sanitized)
        clean_predicted = predicted_view.strip().lower()
        clean_expected = expected_view.strip().lower()

        # E. THE RESPONSE LOGIC
        if clean_predicted == clean_expected and clean_predicted != 'noise_objects':
            # Process and return image string
            enhanced_img = apply_enhancements(image_rgb)
            enhanced_bgr = cv2.cvtColor(enhanced_img, cv2.COLOR_RGB2BGR)
            _, buffer = cv2.imencode('.png', enhanced_bgr)
            img_base64 = base64.b64encode(buffer).decode('utf-8')

            return {
                "match": "Yes",
                "processed_image": f"data:image/png;base64,{img_base64}"
            }
        else:
            # Did not match or was noise
            return {
                "match": "No",
                "processed_image": None
            }

    except Exception:
        # Silently fail with "No" to keep the UI clean
        return {"match": "No", "processed_image": None}
