# main.py

import os
from io import BytesIO

import gdown
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# ─── CONFIGURATION ──────────────────────────────────────────────────────────────
MODEL_DRIVE_ID   = "1I-JAb8EEJviJvdcOyg9iHMPOHgxD1QJ6"   # your Google Drive file ID
MODEL_LOCAL_PATH = "image_caption_model"                # where the model will be saved
FRONTEND_ORIGINS = ["http://localhost:5173/"]            # adjust if your React app runs elsewhere

# ─── FASTAPI SETUP ──────────────────────────────────────────────────────────────
app = FastAPI()

# enable CORS so your React front end can call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── DOWNLOAD & LOAD MODEL ──────────────────────────────────────────────────────
def download_model():
    if not os.path.exists(MODEL_LOCAL_PATH):
        url = f"https://drive.google.com/uc?id={MODEL_DRIVE_ID}"
        print(f"Downloading model from {url} → {MODEL_LOCAL_PATH}")
        gdown.download(url, MODEL_LOCAL_PATH, quiet=False)

@app.on_event("startup")
def load_caption_model():
    download_model()
    global caption_model
    # adjust this if you saved as a folder (SavedModel) vs .h5
    caption_model = tf.keras.models.load_model(MODEL_LOCAL_PATH)
    caption_model.trainable = False
    print("Model loaded and ready for inference.")

# ─── PREPROCESSING ─────────────────────────────────────────────────────────────
def preprocess_image(jpeg_bytes: bytes) -> tf.Tensor:
    """
    Decode JPEG bytes, resize to 299×299, apply InceptionV3 preprocess, and add batch dim.
    """
    img = tf.io.decode_jpeg(jpeg_bytes, channels=3)                   # [H,W,3]
    img = tf.image.resize(img, [299, 299])                            # [299,299,3]
    img = tf.keras.applications.inception_v3.preprocess_input(img)    # scale to [-1,1]
    return tf.expand_dims(img, axis=0)                                # [1,299,299,3]

# ─── PREDICTION ENDPOINT ───────────────────────────────────────────────────────
@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    # 1) Validate file type
    if image.content_type not in ("image/jpeg", "image/jpg"):
        raise HTTPException(status_code=400, detail="Only JPG files are supported.")
    # 2) Read bytes
    body = await image.read()
    # 3) Preprocess
    try:
        batch = preprocess_image(body)
    except Exception:
        raise HTTPException(status_code=400, detail="Failed to decode or preprocess image.")
    # 4) Inference
    try:
        # if your model has a custom call signature, adjust accordingly
        preds = caption_model(batch, training=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")
    # 5) Post-process output
    # If your model returns a Tensor of strings:
    if isinstance(preds, tf.Tensor) and preds.dtype == tf.string:
        caption = preds.numpy()[0].decode("utf-8")
    else:
        # fallback: convert whatever it returned to string
        caption = str(preds.numpy() if hasattr(preds, "numpy") else preds)
    # 6) Return
    return {"result": caption}

# ─── RUN WITH: uvicorn main:app --reload --host 0.0.0.0 --port 8000 ─────────────
