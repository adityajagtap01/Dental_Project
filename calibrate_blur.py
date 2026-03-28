import cv2
import os
import numpy as np

def calibrate_threshold(dataset_path):
    print(f"🔍 Analyzing your perfect dataset at: {"D:\dataset"}...\n")
    
    blur_scores = []
    
    # Walk through every folder and image in your dataset
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg')):
                img_path = os.path.join(root, file)
                
                # Read image and calculate score
                img = cv2.imread(img_path)
                if img is not None:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    score = cv2.Laplacian(gray, cv2.CV_64F).var()
                    blur_scores.append(score)
    
    if not blur_scores:
        print("❌ Could not find any images. Check your folder path.")
        return

    # Calculate the statistics
    min_score = min(blur_scores)
    avg_score = np.mean(blur_scores)
    max_score = max(blur_scores)
    
    # We set the recommended threshold just slightly below your lowest perfect image
    recommended_threshold = max(5.0, min_score - 5.0) 

    print("📊 --- CALIBRATION RESULTS ---")
    print(f"Total Images Analyzed: {len(blur_scores)}")
    print(f"Sharpest Image Score: {max_score:.2f}")
    print(f"Average Image Score:  {avg_score:.2f}")
    print(f"Blurriest 'Perfect' Image Score: {min_score:.2f}")
    print("-" * 30)
    print(f"✅ RECOMMENDED ACTION:")
    print(f"Open 'quality_check.py' and change 'blur_threshold' to: {recommended_threshold:.2f}")

if __name__ == "__main__":
    # Point this to your dataset folder
    calibrate_threshold(r"D:\dataset")