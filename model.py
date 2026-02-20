from PIL import Image, ImageStat, ImageFilter, ImageOps
import io, os, numpy as np, joblib, exifread

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "image_model_improved.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "image_scaler.pkl")

# Load model
try:
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded")
except Exception as e:
    print("❌ Model load error:", e)
    model = None

# Load scaler
try:
    scaler = joblib.load(SCALER_PATH)
    print("✅ Scaler loaded")
except Exception as e:
    print("❌ Scaler load error:", e)
    scaler = None


def extract_features(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    img = ImageOps.exif_transpose(img).convert("RGB")

    gray = img.convert("L")
    contrast = ImageStat.Stat(gray).stddev[0]

    edges = gray.filter(ImageFilter.FIND_EDGES)
    sharpness = ImageStat.Stat(edges).mean[0]

    hist = img.histogram()
    r_mean = np.mean(hist[0:256])
    g_mean = np.mean(hist[256:512])
    b_mean = np.mean(hist[512:768])

    w, h = img.size
    aspect = w / h if h else 1.0

    try:
        tags = exifread.process_file(io.BytesIO(image_bytes), details=False)
        has_exif = 1 if len(tags) > 0 else 0
    except:
        has_exif = 0

    return np.array([[contrast, sharpness, r_mean, g_mean, b_mean, aspect, has_exif]])


def analyze_image_bytes(image_bytes, filename="unknown"):
    if model is None:
        return {"result": "Model not loaded", "confidence": 0.0}

    try:
        X = extract_features(image_bytes)

        if scaler is not None:
            X = scaler.transform(X)

        # Predict
        pred = int(model.predict(X)[0])              # 0 = AI, 1 = REAL
        probs = model.predict_proba(X)[0]

        ai_prob = float(probs[0])
        real_prob = float(probs[1])
        confidence = max(ai_prob, real_prob)

        # Final label logic
        if confidence < 0.60:
            label = "Uncertain 🤔"
        elif pred == 0:
            label = "AI Generated 🤖"
        elif pred == 1:
            label = "Real Image 📸"
        else:
            label = "Uncertain 🤔"

        return {
            "result": label,
            "confidence": round(confidence, 2)
        }

    except Exception as e:
        print("❌ MODEL ERROR:", e)
        return {"result": "Server error", "confidence": 0.0}
