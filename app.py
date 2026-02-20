from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
from model import analyze_image_bytes

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    html_path = os.path.join(os.path.dirname(__file__), "frontend", "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.post("/detect")
async def detect_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        result = analyze_image_bytes(contents, filename=file.filename)

        # 🔥 GUARANTEE JSON FORMAT
        if "result" not in result:
            return JSONResponse(
                {"result": "Processing failed", "confidence": 0.0},
                status_code=200
            )

        return JSONResponse(result, status_code=200)

    except Exception as e:
        print("❌ API Error:", e)
        return JSONResponse(
            {"result": "Server error", "confidence": 0.0},
            status_code=200
        )

