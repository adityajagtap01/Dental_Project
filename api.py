from fastapi import FastAPI, File, UploadFile, Form
import cv2
import numpy as np
import tensorflow as tf
import os
import json
import time
from quality_check import check_image_quality

# --- 1. SETUP THE SERVER ---
app = FastAPI(title="Dental AI Backend")
MODEL_PATH = 'dental_view_model_v2.keras'
OUTPUT_FOLDER = 'processed_data'
CONFIDENCE_THRESHOLD = 0.40 

CLASSES = [
    'Lower Front View', 'Lower Left View', 'Lower Occlusal View', 
    'Lower Right View', 'noise_objects', 'Upper Front View', 
    'Upper Left View', 'Upper Occlusal View', 'Upper Right View'
]

print("🧠 Loading AI Model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Ready!")

# --- 2. IMAGE PROCESSING FUNCTIONS ---
def apply_enhancements(image):
    img_resized = cv2.resize(image, (1024, 1024))
    lab = cv2.cvtColor(img_resized, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

def save_data(image, view_name, patient_id):
    save_path = os.path.join(OUTPUT_FOLDER, patient_id)
    os.makedirs(save_path, exist_ok=True)
    timestamp = int(time.time())
    safe_view = view_name.replace(" ", "_")
    
    img_path = os.path.join(save_path, f"{patient_id}_{safe_view}_{timestamp}.png")
    cv2.imwrite(img_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
    json_path = os.path.join(save_path, f"{patient_id}_{safe_view}_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump({"patient_id": patient_id, "view": view_name}, f)

# --- 3. THE MAGIC ENDPOINT (Where the UI talks to you) ---
@app.post("/analyze-view/")
async def analyze_view(
    file: UploadFile = File(...), 
    patient_id: str = Form(...),
    expected_view: str = Form(...) # The UI tells us what it's trying to capture
):
    try:
        # A. Read the incoming image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # B. OpenCV Blur/Light Check
        is_valid, message, metrics = check_image_quality(image)
        if not is_valid:
            return {"status": "FAIL", "reason": "QUALITY_POOR", "message": message}

        # C. AI Classification
        img_for_ai = cv2.resize(image, (224, 224))
        img_array = np.expand_dims(img_for_ai, axis=0) / 255.0
        preds = model.predict(img_array)
        confidence = float(np.max(preds))
        predicted_view = CLASSES[np.argmax(preds)]

        # D. The "Match" Check (This makes the 8-image loop work!)
        if predicted_view == 'noise_objects':
             return {"status": "FAIL", "reason": "NOISE", "message": "Please point camera at teeth."}
        
        if predicted_view != expected_view:
             return {"status": "FAIL", "reason": "WRONG_ANGLE", "message": f"Expected {expected_view}, but saw {predicted_view}."}

        # E. If everything is perfect: Process, Save, and tell UI to move to the next step
        enhanced_img = apply_enhancements(image)
        save_data(enhanced_img, predicted_view, patient_id)

        return {"status": "SUCCESS", "message": "View saved! Move to next."}

    except Exception as e:
        return {"status": "ERROR", "message": str(e)}