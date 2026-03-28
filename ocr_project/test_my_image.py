"""
Easy OCR test - Just change the IMAGE_PATH below to your image!
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from backend.inference.predict import OCRPredictor

# ✏️ CHANGE THIS to your image path
IMAGE_PATH = r"C:\Users\Satyaprakash\Downloads\WhatsApp Image 2026-03-07 at 11.34.17 PM.jpeg"
# Example: r"C:\Users\YourName\Desktop\handwriting.jpg"

# Check if file exists
if not os.path.exists(IMAGE_PATH):
    print(f"❌ Image not found: {IMAGE_PATH}")
    print("\n📝 Instructions:")
    print("1. Save your handwriting image somewhere")
    print("2. Open this file in VS Code")
    print("3. Change IMAGE_PATH to your image location")
    print("4. Run this script again")
    exit()

# Load model
print("Loading OCR model...")
predictor = OCRPredictor("saved_models/crnn_iam_v1_inference.keras")

# Extract text
print(f"\nReading image: {os.path.basename(IMAGE_PATH)}")
print("Running OCR...\n")

text = predictor.predict(IMAGE_PATH)

# Show result
print("="*60)
print("  📝 EXTRACTED TEXT:")
print(f'  "{text}"')
print("="*60)

print("\n✅ Done!")
