# app.py

import io, logging
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from openai import OpenAI
import httpx
from PIL import Image

# ——— Logging ———
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv() 
# ——— Instantiate the new OpenAI client ———
client = OpenAI()  # pulls api_key from OPENAI_API_KEY env var

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/generate-image")
async def generate_image(payload: dict):
    prompt = payload.get("prompt")
    if not prompt:
        raise HTTPException(400, "No prompt provided")

    # 1) Generate image URL with the new images.generate interface
    try:
        resp = client.images.generate(
            model="dall-e-2",
            prompt=prompt,
            n=1,
            size="512x512",
            response_format="url"
        )
        image_url = resp.data[0].url
        logger.info(f"Image URL: {image_url}")
    except Exception as e:
        logger.exception("OpenAI image generation failed")
        return JSONResponse(
            status_code=502,
            content={"error": "OpenAI generation error", "detail": str(e)}
        )

    # 2) Fetch the PNG bytes
    try:
        async with httpx.AsyncClient() as http:
            r = await http.get(image_url, timeout=10.0)
            r.raise_for_status()
            png_bytes = r.content
    except Exception as e:
        logger.exception("Failed to fetch generated image")
        return JSONResponse(
            status_code=502,
            content={"error": "Image fetch failed", "detail": str(e)}
        )

    # 3) Convert PNG → JPEG
    try:
        img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        buf.seek(0)
    except Exception as e:
        logger.exception("JPEG conversion error")
        return JSONResponse(
            status_code=500,
            content={"error": "JPEG conversion failed", "detail": str(e)}
        )

    # 4) Stream back to client
    return StreamingResponse(buf, media_type="image/jpeg")
