# app.py
import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image
from pathlib import Path
import config
import os

# โหลดโมเดล Transfer Learning ที่เทรนเสร็จแล้ว
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model/transfer_model.h5"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Please run train.py first.")

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

# ชื่อคลาสเรียงตามตัวอักษรโฟลเดอร์ (bird, cat, dog)
CLASS_NAMES = ['Bird', 'Cat', 'Dog']

def predict_animal(image):
    if image is None:
        raise ValueError("Please upload an image.")
    
    # ปรับขนาดและแปลงข้อมูลให้ตรงกับที่โมเดลต้องการ
    img = Image.fromarray(image).convert('RGB')
    img = img.resize(config.IMG_SIZE)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    
    # ทำนายผล
    probabilities = model.predict(arr)[0]
    scores = {CLASS_NAMES[i]: float(probabilities[i]) for i in range(len(CLASS_NAMES))}
    return scores

# สร้างหน้าเว็บ
app = gr.Interface(
    fn=predict_animal,
    inputs=gr.Image(type="numpy", label="Upload an Image"),
    outputs=gr.Label(num_top_classes=3, label="Prediction"),
    title="Paws & Wings AI: Animal Classifier",
    description="อัปโหลดรูป นก, แมว หรือ สุนัข เพื่อให้ AI ทำนายผล"
)

if __name__ == "__main__":
    app.launch( 
        server_name=os.getenv("HOST", "127.0.0.1"),
        server_port=int(os.getenv("PORT", "7860"))
    )