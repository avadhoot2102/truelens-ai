import os
import numpy as np
from PIL import Image, ImageStat, ImageFilter
from sklearn.ensemble import RandomForestClassifier
import joblib

DATASET_PATH = "dataset"

def extract_features(image_path):
    try:
        image = Image.open(image_path).convert("L")
        stat = ImageStat.Stat(image)
        contrast = stat.stddev[0]
        edges = image.filter(ImageFilter.FIND_EDGES)
        sharpness = ImageStat.Stat(edges).mean[0]
        return [contrast, sharpness]
    except:
        return [0, 0]

X, y = [], []
for label, folder in enumerate(["real", "ai"]):
    path = os.path.join(DATASET_PATH, folder)
    for img_file in os.listdir(path):
        img_path = os.path.join(path, img_file)
        features = extract_features(img_path)
        X.append(features)
        y.append(label)

X, y = np.array(X), np.array(y)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)


model = RandomForestClassifier(n_estimators=200)
model.fit(X, y)

joblib.dump(model, "image_model.pkl")
print("✅ Improved Model trained and saved as image_model.pkl")
