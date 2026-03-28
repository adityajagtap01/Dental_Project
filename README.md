🦷 Dental AI Gatekeeper - Backend API

The Dental AI Gatekeeper API acts as an intelligent backend system that validates dental images before they are stored or passed to downstream diagnostic models.

It ensures that only:

✅ High-quality images
✅ Correct dental angles
✅ Valid dental content

are accepted into the system.

📌 Overview

This API sits between:

Frontend UI (Image Capture Interface)
Diagnostic AI System (Module 2)

It performs:

Image Quality Validation (OpenCV)
Angle Classification (MobileNetV2 Model)
Image Enhancement (CLAHE)
Secure Storage
🔄 Workflow

For each dental view:

📸 User captures an image from the frontend
📤 UI sends request to API
🧠 API performs:
Quality Check (blur/brightness)
AI Classification (correct view or not)
✨ If valid → Image enhanced & saved
📩 API responds with JSON:
SUCCESS → Move to next view
FAIL → Retake image
🛠️ Tech Stack
Backend Framework: FastAPI
Server: Uvicorn
Computer Vision: OpenCV
ML Model: TensorFlow (MobileNetV2)
Data Processing: NumPy
⚙️ Local Setup
✅ Prerequisites
Python 3.9+

Pre-trained model file:

dental_view_model_v2.keras

(Place it in the root directory)

📦 Installation
pip install fastapi uvicorn python-multipart opencv-python numpy tensorflow
📁 Directory Setup

Create a folder for storing processed images:

mkdir processed_data
▶️ Running the Server
uvicorn api:app --reload

Server will run at:

http://127.0.0.1:8000
📡 API Endpoint
🔹 POST /analyze-view/
📌 Headers
Content-Type: multipart/form-data
📥 Request Body (Form Data)
Key	Type	Description
file	File	Image (.jpg, .png)
patient_id	String	Unique patient ID
expected_view	String	Expected dental angle
🦷 Accepted Views
Lower Front View
Lower Left View
Lower Occlusal View
Lower Right View
Upper Front View
Upper Left View
Upper Occlusal View
Upper Right View

⚠️ Important: Case-sensitive and must match exactly.

📤 API Responses
✅ Success
{
  "status": "SUCCESS",
  "message": "View saved! Move to next."
}
❌ Failure - Poor Quality
{
  "status": "FAIL",
  "reason": "QUALITY_POOR",
  "message": "Image is too blurry. Hold the camera steady."
}
❌ Failure - Wrong Angle
{
  "status": "FAIL",
  "reason": "WRONG_ANGLE",
  "message": "Expected Lower Front View, but saw Upper Front View."
}
🧪 Testing the API (No Code Required)

FastAPI provides a built-in UI for testing.

Steps:
Start the server

Open browser:

http://127.0.0.1:8000/docs
Select POST /analyze-view/
Click Try it out
Upload image + inputs
Click Execute
📂 Output Storage

All validated and enhanced images are stored in:

/processed_data
🚀 Key Features
🔍 Automated image quality detection
🧠 AI-based dental view classification
✨ Medical-grade contrast enhancement (CLAHE)
⚡ FastAPI high-performance backend
🔐 Clean validation layer before diagnosis
📌 Notes for Frontend Developers
Always send data as multipart/form-data
Validate status field in response:
SUCCESS → Move to next step
FAIL → Show message and retry
Ensure expected_view matches exactly
🤝 Future Improvements
Add authentication layer
Cloud storage integration (AWS/GCP)
Real-time feedback overlay for users
Multi-language support for UI messages

If you want, I can also:

🔥 Generate api.py structure for this
🎯 Add system architecture diagram
📊 Help you integrate this with your frontend

Just tell me 👍
