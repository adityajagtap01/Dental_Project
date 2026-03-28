# 🦷 Dental AI Gatekeeper - Backend API

An intelligent backend API that validates, classifies, and enhances dental images before storing or forwarding them to diagnostic AI systems.

---

## 📌 Overview

This API acts as a **Gatekeeper** between the frontend and diagnostic AI.

It ensures only:
- ✅ High-quality images  
- ✅ Correct dental angles  
- ✅ Valid dental content  

are processed further.

---

## 🔄 Workflow

1. 📸 Capture dental image (Frontend)
2. 📤 Send image + patient_id + expected_view
3. 🔍 API checks:
   - Image Quality (blur/brightness)
   - Correct View (AI Model)
4. ✨ If valid → Enhance (CLAHE) & Save
5. 📩 Response:
   - `SUCCESS` → Next step
   - `FAIL` → Retake image

---

## 🛠️ Tech Stack

- FastAPI
- Uvicorn
- OpenCV
- TensorFlow (MobileNetV2)
- NumPy

---

## ⚙️ Setup

### 1. Install Dependencies
```bash
pip install fastapi uvicorn python-multipart opencv-python numpy tensorflow
2. Add Model File

Place in root directory:

dental_view_model_v2.keras
3. Create Storage Folder
mkdir processed_data

▶️ Run Server
uvicorn api:app --reload

Server runs at:

http://127.0.0.1:8000
📡 API Endpoint
POST /analyze-view/

Content-Type: multipart/form-data

Request Fields
Field	Type
file	File
patient_id	String
expected_view	String
Accepted Views
Lower Front View
Lower Left View
Lower Occlusal View
Lower Right View
Upper Front View
Upper Left View
Upper Occlusal View
Upper Right View

📤 Responses
✅ Success
{
  "status": "SUCCESS",
  "message": "View saved! Move to next."
}
❌ Quality Issue
{
  "status": "FAIL",
  "reason": "QUALITY_POOR",
  "message": "Image is too blurry."
}
❌ Wrong Angle
{
  "status": "FAIL",
  "reason": "WRONG_ANGLE",
  "message": "Incorrect dental view."
}
🧪 Testing

Open Swagger UI:

http://127.0.0.1:8000/docs

📂 Project Structure
project/
├── api.py
├── dental_view_model_v2.keras
├── processed_data/
└── README.md

🚀 Key Features
Image quality validation (OpenCV)
AI-based dental view classification
CLAHE image enhancement
FastAPI high-performance backend
📌 Notes
Use multipart/form-data
expected_view must match exactly
Handle API response (SUCCESS / FAIL) properly

