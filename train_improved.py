import os
import numpy as np
from PIL import Image, ImageStat, ImageFilter, ImageOps
import exifread
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib

DATASET_DIR = "dataset"   # dataset/ai , dataset/real


def extract_features_from_path(path):
    try:
        img = Image.open(path)
        img = ImageOps.exif_transpose(img)
        img.load()
    except Exception:
        return None

    rgb = img.convert("RGB")
    gray = rgb.convert("L")

    # Texture features
    contrast = ImageStat.Stat(gray).stddev[0]
    edges = gray.filter(ImageFilter.FIND_EDGES)
    sharpness = ImageStat.Stat(edges).mean[0]

    # Color histogram features
    hist = rgb.histogram()
    r_mean = np.mean(hist[0:256])
    g_mean = np.mean(hist[256:512])
    b_mean = np.mean(hist[512:768])

    # Geometry
    w, h = img.size
    aspect = w / h if h else 1.0

    # EXIF presence
    try:
        with open(path, "rb") as f:
            tags = exifread.process_file(f, details=False)
        has_exif = 1 if len(tags) > 0 else 0
    except:
        has_exif = 0

    return [contrast, sharpness, r_mean, g_mean, b_mean, aspect, has_exif]


def load_dataset():
    X, y = [], []

    # ✅ FINAL LABEL MAPPING (LOCK THIS)
    classes = {"real": 1, "ai": 0}


    for cls, label in classes.items():
        folder = os.path.join(DATASET_DIR, cls)
        if not os.path.exists(folder):
            continue

        for fname in os.listdir(folder):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                continue

            path = os.path.join(folder, fname)
            feats = extract_features_from_path(path)
            if feats is None:
                continue

            X.append(feats)
            y.append(label)

    return np.array(X, dtype=float), np.array(y, dtype=int)


if __name__ == "__main__":
    print("📥 Loading dataset...")
    X, y = load_dataset()
    print(f"✅ Loaded {len(y)} images")

    # Safety check
    if len(np.unique(y)) < 2:
        raise ValueError("❌ Dataset must contain BOTH ai and real images")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # ✅ Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("🧠 Training RandomForest model...")
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    print("✅ Accuracy:", accuracy_score(y_test, y_pred))

    # ✅ CORRECT classification report
    print(classification_report(
        y_test,
        y_pred,
        target_names=["AI", "REAL"]
    ))

    # Feature importance (debug / trust)
    print("\n📊 Feature Importance:")
    for name, score in zip(
        ["contrast", "sharpness", "r", "g", "b", "aspect", "exif"],
        model.feature_importances_
    ):
        print(f"{name:12s} : {score:.3f}")

    # Save model + scaler
    joblib.dump(model, "image_model_improved.pkl")
    joblib.dump(scaler, "image_scaler.pkl")
    print("\n💾 Saved:")
    print("   image_model_improved.pkl")
    print("   image_scaler.pkl")
