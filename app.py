import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import os
import json
import time
from quality_check import check_image_quality

# --- CONFIGURATION ---
MODEL_PATH = 'dental_view_model_v2.keras'
OUTPUT_FOLDER = 'processed_data'
CONFIDENCE_THRESHOLD = 0.50

# Must match alphabetical training order exactly!
CLASSES = [
    'Lower Front View', 'Lower Left View', 'Lower Occlusal View', 
    'Lower Right View', 'noise_objects', 'Upper Front View', 
    'Upper Left View', 'Upper Occlusal View', 'Upper Right View'
]

# --- LOAD AI MODEL ---
@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        return None

model = load_model()

# --- HELPER FUNCTIONS ---
def apply_enhancements(image):
    """Applies MedGemma-standard CLAHE enhancement and resizes to 1024x1024."""
    img_resized = cv2.resize(image, (1024, 1024))
    lab = cv2.cvtColor(img_resized, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

def save_data(image, view_name, patient_id):
    """Saves the cleaned image + JSON instructions for Module 2"""
    save_path = os.path.join(OUTPUT_FOLDER, patient_id)
    os.makedirs(save_path, exist_ok=True)
    
    timestamp = int(time.time())
    safe_view = view_name.replace(" ", "_")
    img_filename = f"{patient_id}_{safe_view}_{timestamp}.png"
    json_filename = f"{patient_id}_{safe_view}_{timestamp}.json"
    
    full_img_path = os.path.join(save_path, img_filename)
    cv2.imwrite(full_img_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
    metadata = {
        "patient_id": patient_id,
        "view_detected": view_name,
        "processing": "CLAHE + Resize",
        "medgemma_instruction": f"Analyze this {view_name} for cavities."
    }
    with open(os.path.join(save_path, json_filename), 'w') as f:
        json.dump(metadata, f, indent=4)
        
    return full_img_path

# --- MAIN INTERFACE ---
st.set_page_config(page_title="Dental AI Gatekeeper", page_icon="🦷")
st.title("🦷 Module 1: Intelligent Pre-Processing")
st.write("Upload a raw dental image. The system will Validate, Classify, and Enhance it.")

if model is None:
    st.error(f"⚠️ Could not load AI model. Please ensure '{MODEL_PATH}' is in the same folder as this script.")
    st.stop()

patient_id = st.text_input("Enter Patient ID (e.g., P-101)", "Guest_001")
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # 1. READ IMAGE
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_image = cv2.imdecode(file_bytes, 1)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB) 
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(original_image, caption="Original Raw Input")

    st.markdown("### 🔍 Running Analysis...")
    
    # 2. RULE-BASED QUALITY CHECK (OpenCV)
    is_valid, message, metrics = check_image_quality(original_image)
    
    if not is_valid:
        st.error(f"❌ {message}")
    else:
        st.success(f"✅ Quality Check Passed: {message}")
        
        # 3. AI CLASSIFICATION (MobileNetV2)
        img_for_ai = cv2.resize(original_image, (224, 224))
        img_array = np.expand_dims(img_for_ai, axis=0) / 255.0
        
        preds = model.predict(img_array)
        confidence = np.max(preds)
        predicted_view = CLASSES[np.argmax(preds)]
        
        # 4. FOREIGN OBJECT / CONFIDENCE CHECK
        if predicted_view == 'noise_objects':
            st.error("❌ REJECTED: Invalid Object Detected. Please capture a clear dental view.")
        elif confidence < CONFIDENCE_THRESHOLD:
            st.warning(f"⚠️ REJECTED: Low Confidence ({confidence*100:.1f}%). Is this a clear tooth?")
            st.write(f"Best guess was: {predicted_view}")
        else:
            # 5. ENHANCEMENT & SAVING
            with st.spinner("Applying CLAHE Enhancement & Generating JSON..."):
                enhanced_img = apply_enhancements(original_image)
                saved_path = save_data(enhanced_img, predicted_view, patient_id)
            
            with col2:
                st.image(enhanced_img, caption=f"✅ Enhanced & Ready for MedGemma\n({predicted_view})")
            
            st.success(f"Files saved to: {saved_path}")
            st.json({"View": predicted_view, "Confidence": f"{confidence*100:.2f}%", "Status": "Passed to Module 2"})