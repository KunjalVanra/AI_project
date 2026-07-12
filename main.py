import pandas as pd
import joblib

import waste_management_ml as ml
import waste_management_dl as dl


# =====================================================
# LOAD DATASET
# =====================================================

def load_dataset():

    path = "Waste_Dataset_With_City_10000.csv"

    df = pd.read_csv(path)

    print("\nDataset Loaded Successfully")

    print(df.head())

    return df



# =====================================================
# PREPROCESS DATA
# =====================================================

def preprocess(df):

    print("\nCleaning Dataset...")

    df.drop_duplicates(inplace=True)

    df.fillna(method="ffill", inplace=True)


    columns = [
        "Waste_Type",
        "Material",
        "Toxicity",
        "Recyclable",
        "Energy_Recovery",
        "City"
    ]


    for col in columns:

        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
        )


    return df



# =====================================================
# USER PREDICTION
# =====================================================

def manual_prediction(model):


    print("\n==============================")
    print(" Waste Classification System ")
    print("==============================")


    material = input("Material: ")

    toxicity = input(
        "Toxicity (Low/Medium/High): "
    )

    recyclable = input(
        "Recyclable (Yes/No): "
    )

    energy = input(
        "Energy Recovery (Low/Medium/High): "
    )

    city = input(
        "City: "
    )


    data = pd.DataFrame([{

        "Material": material.strip(),

        "Toxicity": toxicity.strip(),

        "Recyclable": recyclable.strip().title(),

        "Energy_Recovery": energy.strip(),

        "City": city.strip()

    }])


    encoders = joblib.load(
        "encoders.pkl"
    )


    for col, encoder in encoders.items():

        data[col] = encoder.transform(
            data[col]
        )


    data["Recyclable"] = (
        1 if recyclable.strip().title()=="Yes"
        else 0
    )


    data = data[
        [
            "Material",
            "Toxicity",
            "Recyclable",
            "Energy_Recovery",
            "City"
        ]
    ]



    prediction = model.predict(data)[0]


    target_encoder = joblib.load(
        "target_encoder.pkl"
    )


    prediction = target_encoder.inverse_transform(
        [prediction]
    )[0]


    print(
        "\nPredicted Waste Type:",
        prediction
    )


    print("\nRecommendation:")

    print(
        ml.get_recommendation(prediction)
    )



# =====================================================
# MAIN PROGRAM
# =====================================================

if __name__ == "__main__":


    print(
"""
=====================================
 AI SMART WASTE CLASSIFICATION SYSTEM
=====================================
"""
    )


    df = load_dataset()


    df = preprocess(df)



    print(
        "\nTraining Machine Learning Model..."
    )


    (
        model,
        city_encoder,
        _,
        material_encoder,
        toxicity_encoder,
        energy_encoder,
        processed_df

    ) = ml.classify_waste(df)



    encoders = {

        "City": city_encoder,

        "Material": material_encoder,

        "Toxicity": toxicity_encoder,

        "Energy_Recovery": energy_encoder

    }


    joblib.dump(
        encoders,
        "encoders.pkl"
    )


    print(
        "\nModel Training Completed"
    )



    while True:


        print(
"""
==============================
MENU
==============================

1. Predict Waste Type

2. Exit

"""
        )


        choice = input(
            "Enter choice: "
        )


        if choice == "1":

            manual_prediction(
                model
            )


        elif choice == "2":

            print(
                "Thank You"
            )

            break


        else:

            print(
                "Invalid Choice"
            )