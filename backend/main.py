import os
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# ─── CONFIGURATION ──────────────────────────────────────────────────────────────
MODEL_DIR         = "image_caption_model"   # directory containing SavedModel
FRONTEND_ORIGINS  = ["http://localhost:3000"]

# ─── FASTAPI SETUP ──────────────────────────────────────────────────────────────
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── LOAD SAVEDMODEL & INFERENCE SIGNATURE ────────────────────────────────────────
@app.on_event("startup")
def load_caption_model():
    global caption_model
    # Ensure model directory exists
    if not os.path.isdir(MODEL_DIR):
        raise RuntimeError(f"Model directory '{MODEL_DIR}' not found.")

    try:
        # Load the TensorFlow SavedModel
        saved = tf.saved_model.load(MODEL_DIR)
        # Grab the default serving function for inference
        signature = saved.signatures.get('serving_default')
        if signature is None:
            # fallback to first available signature
            signature = list(saved.signatures.values())[0]
        caption_model = signature
    except Exception as e:
        raise RuntimeError(
            f"Failed to load SavedModel at '{MODEL_DIR}': {e}"
        )
    print(f"✅ SavedModel loaded successfully with signature '{caption_model.name}'.")

# ─── PREPROCESSING ───────────────────────────────────────────────────────────────
def preprocess_image(jpeg_bytes: bytes) -> tf.Tensor:
    img = tf.io.decode_jpeg(jpeg_bytes, channels=3)                   # [H,W,3]
    img = tf.image.resize(img, [299, 299])                            # [299,299,3]
    img = tf.keras.applications.inception_v3.preprocess_input(img)    # scale to [-1,1]
    return tf.expand_dims(img, axis=0)                                # [1,299,299,3]

# ─── PREDICTION ENDPOINT ─────────────────────────────────────────────────────────
@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    # 1) Validate file type
    if image.content_type not in ("image/jpeg", "image/jpg"):
        raise HTTPException(400, "Only JPG files are supported.")

    # 2) Read bytes
    body = await image.read()

    # 3) Preprocess
    try:
        batch = preprocess_image(body)
    except Exception:
        raise HTTPException(400, "Failed to decode or preprocess image.")

    # 4) Inference via SavedModel signature
    try:
        preds_dict = caption_model(batch)
        # get first tensor output
        preds = list(preds_dict.values())[0]
    except Exception as e:
        raise HTTPException(500, f"Inference error: {e}")

    # 5) Post-process
    if isinstance(preds, tf.Tensor) and preds.dtype == tf.string:
        caption = preds.numpy()[0].decode("utf-8")
    else:
        caption = str(preds.numpy().tolist())

    return {"result": caption}

# ─── RUN: uvicorn main:app --reload --host 0.0.0.0 --port 8000 ─────────────────────
