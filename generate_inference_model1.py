import os
import sys
import tensorflow as tf

# Set working directory to script's parent (ocr_project)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(script_dir, '..'))
# Ensure working directory is ocr_project
os.chdir(project_dir)
# Add ocr_project and its parent to sys.path for robust imports
sys.path.insert(0, project_dir)
sys.path.insert(0, os.path.dirname(project_dir))


from backend.models.crnn_model import build_crnn_model, build_inference_model
from backend.preprocessing.image_processor import IMG_HEIGHT, IMG_WIDTH
from backend.utils.char_map import get_num_classes

SAVE_DIR = os.path.join(project_dir, "saved_models")
best_weights = os.path.join(SAVE_DIR, "crnn_camera_local_best.weights.h5")
inference_path = os.path.join(SAVE_DIR, "crnn_camera_local_inference.keras")

# Build model architecture with explicit parameters
num_classes = get_num_classes()
model = build_crnn_model(
    img_height=IMG_HEIGHT,
    img_width=IMG_WIDTH,
    num_classes=num_classes,
    lstm_units=256,
    dropout_rate=0.25
)

# Load best weights
if os.path.exists(best_weights):
    model.load_weights(best_weights)
    print(f"Loaded best weights from {best_weights}")
else:
    raise FileNotFoundError(f"Best weights not found: {best_weights}")

# Build inference model
inference_model = build_inference_model(model)

# Save inference model
inference_model.save(inference_path)
print(f"Inference model saved: {inference_path}")
