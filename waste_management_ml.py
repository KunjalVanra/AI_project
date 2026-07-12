import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)



# =====================================================
# Recommendation System
# =====================================================

def get_recommendation(category):

    recommendations = {

        "E-Waste":
            "Dispose through authorized electronic waste recycling centers.",

        "Plastic":
            "Send plastic waste to recycling facilities.",

        "Paper":
            "Recycle paper waste through paper recycling units.",

        "Organic":
            "Use composting or organic waste processing.",

        "Metal":
            "Send metal waste for recovery and recycling."

    }

    return recommendations.get(
        category,
        "Follow your local waste management guidelines."
    )



# =====================================================
# Create Category (optional)
# =====================================================

def create_category(df):

    return df



# =====================================================
# Train Model
# =====================================================

def classify_waste(df):


    print("\nPreparing Dataset...\n")


    df = create_category(df)



    # -----------------------------
    # SAVE ORIGINAL TARGET
    # -----------------------------

    target_encoder = LabelEncoder()


    y = target_encoder.fit_transform(
        df["Waste_Type"]
    )


    print("\nWaste Type Mapping:")

    for i, name in enumerate(target_encoder.classes_):

        print(i, "=", name)



    joblib.dump(
        target_encoder,
        "target_encoder.pkl"
    )



    # -----------------------------
    # INPUT ENCODERS
    # -----------------------------


    city_encoder = LabelEncoder()

    material_encoder = LabelEncoder()

    toxicity_encoder = LabelEncoder()

    energy_encoder = LabelEncoder()



    X = df[
        [
            "Material",
            "Toxicity",
            "Recyclable",
            "Energy_Recovery",
            "City"
        ]
    ].copy()



    X["City"] = city_encoder.fit_transform(
        X["City"]
    )


    X["Material"] = material_encoder.fit_transform(
        X["Material"]
    )


    X["Toxicity"] = toxicity_encoder.fit_transform(
        X["Toxicity"]
    )


    X["Energy_Recovery"] = energy_encoder.fit_transform(
        X["Energy_Recovery"]
    )


    X["Recyclable"] = X["Recyclable"].map(
        {
            "Yes":1,
            "No":0
        }
    )



    # -----------------------------
    # Split Dataset
    # -----------------------------


    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=0.20,

        random_state=42,

        stratify=y

    )



    print("\nTraining Random Forest...\n")



    parameters = {

        "n_estimators":[100,200],

        "max_depth":[10,15,None]

    }



    rf = RandomForestClassifier(
        random_state=42
    )



    model = GridSearchCV(

        rf,

        parameters,

        cv=5,

        scoring="accuracy"

    )



    model.fit(
        X_train,
        y_train
    )



    best_model = model.best_estimator_



    predictions = best_model.predict(
        X_test
    )



    # -----------------------------
    # Performance
    # -----------------------------


    accuracy = accuracy_score(
        y_test,
        predictions
    )


    precision = precision_score(
        y_test,
        predictions,
        average="weighted"
    )


    recall = recall_score(
        y_test,
        predictions,
        average="weighted"
    )


    f1 = f1_score(
        y_test,
        predictions,
        average="weighted"
    )



    print("="*50)

    print("RANDOM FOREST PERFORMANCE")

    print("="*50)


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



    # Cross Validation

    scores = cross_val_score(

        best_model,

        X,

        y,

        cv=5

    )


    print(
        f"\nCross Validation Accuracy: {scores.mean()*100:.2f}%"
    )



    # -----------------------------
    # Confusion Matrix
    # -----------------------------


    cm = confusion_matrix(
        y_test,
        predictions
    )


    plt.figure(
        figsize=(7,6)
    )


    sns.heatmap(

        cm,

        annot=True,

        fmt="d",

        cmap="Blues"

    )


    plt.title(
        "Waste Classification Confusion Matrix"
    )


    plt.xlabel(
        "Predicted"
    )


    plt.ylabel(
        "Actual"
    )


    plt.savefig(
        "confusion_matrix.png"
    )


    plt.close()



    # Feature importance


    importance = pd.Series(

        best_model.feature_importances_,

        index=X.columns

    )


    importance.sort_values().plot(

        kind="barh",

        figsize=(8,5),

        color="green"

    )


    plt.title(
        "Feature Importance"
    )


    plt.tight_layout()


    plt.savefig(
        "feature_importance.png"
    )


    plt.close()



    # -----------------------------
    # Save Files
    # -----------------------------


    joblib.dump(
        best_model,
        "waste_classifier.pkl"
    )


    joblib.dump(
        {
        "City":city_encoder,
        "Material":material_encoder,
        "Toxicity":toxicity_encoder,
        "Energy_Recovery":energy_encoder
        },

        "encoders.pkl"

    )



    print(
        "\nModel Saved Successfully."
    )



    return (

        best_model,

        city_encoder,

        None,

        material_encoder,

        toxicity_encoder,

        energy_encoder,

        X

    )