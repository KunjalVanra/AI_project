import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2, preprocess_input, decode_predictions
)
from tensorflow.keras.preprocessing import image


# ================= DEEP LEARNING MODEL =================
def optimize_handling(df):
    """
    ANN model to simulate pattern detection
    for waste handling optimization
    """

    # Encode categorical columns
    le_waste = LabelEncoder()
    le_toxicity = LabelEncoder()
    le_city = LabelEncoder()
    le_category = LabelEncoder()

    df["waste_type_encoded"] = le_waste.fit_transform(df["waste_type"])
    df["toxicity_level_encoded"] = le_toxicity.fit_transform(df["toxicity_level"])
    df["city_encoded"] = le_city.fit_transform(df["city"])
    df["category_encoded"] = le_category.fit_transform(df["category"])

    # Input features (MATCH CSV)
    X = df[
        [
            "waste_type_encoded",
            "toxicity_level_encoded",
            "city_encoded",
            "recyclability"
        ]
    ]

    y = df["category_encoded"]

    # Normalize features
    X = (X - X.mean()) / X.std()

    # ANN Model
    model = Sequential([
        Dense(16, activation="relu", input_shape=(X.shape[1],)),
        Dense(8, activation="relu"),
        Dense(len(df["category"].unique()), activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(X, y, epochs=10, batch_size=32, verbose=0)

    loss, accuracy = model.evaluate(X, y, verbose=0)
    print(f"ANN Model Accuracy (Pattern Detection): {accuracy:.2f}")

    return model


# ================= IMAGE ANALYSIS =================
def analyze_image(image_path):
    """
    Uses MobileNetV2 to identify object
    and map it to waste type & toxicity
    """
    try:
        print("Loading image recognition model...")
        image_path = image_path.strip().strip('"').strip("'")
        model = MobileNetV2(weights="imagenet")

        img = image.load_img(image_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        preds = model.predict(x, verbose=0)
        decoded = decode_predictions(preds, top=1)[0][0]

        label = decoded[1].lower()
        confidence = decoded[2]

        print(f"Detected: {label} ({confidence*100:.2f}%)")

        # Mapping detected object to waste info
        if any(k in label for k in ["laptop", "phone", "keyboard", "computer"]):
            return {"waste_type": "E-waste", "toxicity": "high"}

        elif any(k in label for k in ["bottle", "plastic", "container"]):
            return {"waste_type": "Plastic", "toxicity": "low"}

        elif any(k in label for k in ["paper", "newspaper", "book"]):
            return {"waste_type": "Paper", "toxicity": "low"}

        else:
            return {"waste_type": "Plastic", "toxicity": "medium"}

    except Exception as e:
        print("Image processing error:", e)
        return None
    