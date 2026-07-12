import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    BatchNormalization
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)


# ======================================================
# CREATE CATEGORY RECOMMENDATION
# ======================================================

def waste_recommendation(waste_type):

    recommendations = {

        "Paper":
        "Recycle through paper recycling facilities.",

        "Plastic":
        "Send recyclable plastic to a plastic recycling center.",

        "Organic":
        "Process using composting or organic waste treatment.",

        "Metal":
        "Send metal waste for metal recovery and recycling.",

        "E-Waste":
        "Dispose through authorized electronic waste collection centers."

    }

    return recommendations.get(
        waste_type,
        "Follow local waste management guidelines."
    )



# ======================================================
# PREPARE DATA
# ======================================================

def prepare_data(df):

    data = df.copy()


    encoders = {}

    categorical_columns = [
        "Waste_Type",
        "Material",
        "Toxicity",
        "Energy_Recovery",
        "City"
    ]


    for column in categorical_columns:

        encoder = LabelEncoder()

        data[column] = encoder.fit_transform(
            data[column]
        )

        encoders[column] = encoder



    # Convert recyclable

    data["Recyclable"] = data["Recyclable"].map(
        {
            "Yes":1,
            "No":0
        }
    )


    X = data[
        [
            "Material",
            "Toxicity",
            "Recyclable",
            "Energy_Recovery",
            "City"
        ]
    ]


    y = data["Waste_Type"]



    target_encoder = LabelEncoder()

    y = target_encoder.fit_transform(y)


    scaler = StandardScaler()

    X = scaler.fit_transform(X)



    return (
        X,
        y,
        encoders,
        target_encoder,
        scaler
    )



# ======================================================
# BUILD ANN MODEL
# ======================================================

def build_ann(input_shape, classes):


    model = Sequential()


    model.add(
        Dense(
            128,
            activation="relu",
            input_shape=(input_shape,)
        )
    )

    model.add(
        BatchNormalization()
    )

    model.add(
        Dropout(0.3)
    )


    model.add(
        Dense(
            64,
            activation="relu"
        )
    )

    model.add(
        BatchNormalization()
    )

    model.add(
        Dropout(0.3)
    )


    model.add(
        Dense(
            32,
            activation="relu"
        )
    )


    model.add(
        Dense(
            classes,
            activation="softmax"
        )
    )


    model.compile(

        optimizer="adam",

        loss="sparse_categorical_crossentropy",

        metrics=[
            "accuracy"
        ]

    )


    return model



# ======================================================
# TRAIN DEEP LEARNING MODEL
# ======================================================

def train_ann(df):


    print("\nPreparing ANN Dataset...\n")


    X,y,encoders,target_encoder,scaler = prepare_data(df)



    X_train,X_test,y_train,y_test = train_test_split(

        X,
        y,

        test_size=0.2,

        random_state=42,

        stratify=y

    )



    model = build_ann(

        X_train.shape[1],

        len(np.unique(y))

    )



    callbacks=[


        EarlyStopping(

            monitor="val_loss",

            patience=15,

            restore_best_weights=True

        ),



        ReduceLROnPlateau(

            monitor="val_loss",

            factor=0.2,

            patience=5

        ),



        ModelCheckpoint(

            "ann_waste_model.keras",

            save_best_only=True

        )

    ]



    history=model.fit(

        X_train,

        y_train,

        validation_data=(X_test,y_test),

        epochs=100,

        batch_size=32,

        callbacks=callbacks,

        verbose=1

    )



    # Prediction

    predictions=model.predict(
        X_test
    )


    predictions=np.argmax(
        predictions,
        axis=1
    )



    accuracy=accuracy_score(
        y_test,
        predictions
    )


    precision=precision_score(

        y_test,

        predictions,

        average="weighted"

    )


    recall=recall_score(

        y_test,

        predictions,

        average="weighted"

    )


    f1=f1_score(

        y_test,

        predictions,

        average="weighted"

    )



    print("\n==============================")

    print("ANN PERFORMANCE")

    print("==============================")

    print(
        f"Accuracy : {accuracy*100:.2f}%"
    )

    print(
        f"Precision: {precision*100:.2f}%"
    )

    print(
        f"Recall   : {recall*100:.2f}%"
    )

    print(
        f"F1 Score : {f1*100:.2f}%"
    )



    print(
        "\nClassification Report\n"
    )

    print(
        classification_report(
            y_test,
            predictions
        )
    )



    # Accuracy Graph

    plt.figure(figsize=(8,5))


    plt.plot(
        history.history["accuracy"],
        label="Training Accuracy"
    )


    plt.plot(
        history.history["val_accuracy"],
        label="Validation Accuracy"
    )


    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")

    plt.legend()

    plt.title(
        "ANN Accuracy Curve"
    )

    plt.savefig(
        "ann_accuracy.png"
    )

    plt.close()



    # Loss Graph

    plt.figure(figsize=(8,5))


    plt.plot(
        history.history["loss"],
        label="Training Loss"
    )


    plt.plot(
        history.history["val_loss"],
        label="Validation Loss"
    )


    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.legend()


    plt.title(
        "ANN Loss Curve"
    )


    plt.savefig(
        "ann_loss.png"
    )

    plt.close()



    # Confusion Matrix

    cm=confusion_matrix(

        y_test,

        predictions

    )


    plt.figure(figsize=(7,6))


    sns.heatmap(

        cm,

        annot=True,

        fmt="d",

        cmap="Greens"

    )


    plt.title(
        "ANN Confusion Matrix"
    )


    plt.savefig(
        "ann_confusion_matrix.png"
    )


    plt.close()



    # Save preprocessing objects

    joblib.dump(
        encoders,
        "ann_encoders.pkl"
    )


    joblib.dump(
        target_encoder,
        "ann_target_encoder.pkl"
    )


    joblib.dump(
        scaler,
        "ann_scaler.pkl"
    )


    print(
        "\nANN Model Training Completed."
    )


    return model