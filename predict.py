import pandas as pd
import joblib

import waste_management_ml as ml


# =====================================================
# LOAD SAVED MODEL
# =====================================================

def load_model():

    try:

        model = joblib.load(
            "waste_classifier.pkl"
        )

        encoders = joblib.load(
            "encoders.pkl"
        )


        print(
            "\nModel Loaded Successfully"
        )


        return model, encoders


    except Exception as e:

        print(
            "\nModel Loading Error:",
            e
        )

        print(
            "Please train the model first using main.py"
        )

        exit()



# =====================================================
# USER INPUT
# =====================================================

def get_user_input():


    print(
        """
================================
 Waste Type Prediction System
================================
"""
    )


    material=input(
        "Enter Material: "
    )


    toxicity=input(
        "Enter Toxicity (Low/Medium/High): "
    )


    recyclable=input(
        "Recyclable (Yes/No): "
    )


    energy=input(
        "Energy Recovery (Low/Medium/High): "
    )


    city=input(
        "Enter City: "
    )



    data=pd.DataFrame([{


        "Material":material,

        "Toxicity":toxicity,

        "Recyclable":recyclable,

        "Energy_Recovery":energy,

        "City":city


    }])


    return data




# =====================================================
# PREPROCESS INPUT
# =====================================================

def preprocess_input(data,encoders):


    for column,encoder in encoders.items():

        if column!="Waste_Type":

            data[column]=encoder.transform(
                data[column]
            )



    data["Recyclable"]=data["Recyclable"].map(

        {
            "Yes":1,
            "No":0
        }

    )


    data=data[

        [

        "Material",

        "Toxicity",

        "Recyclable",

        "Energy_Recovery",

        "City"

        ]

    ]



    return data




# =====================================================
# PREDICT
# =====================================================

def predict():


    model,encoders=load_model()



    user_data=get_user_input()



    processed_data=preprocess_input(

        user_data,

        encoders

    )



    prediction=model.predict(

        processed_data

    )[0]



    print(
        "\n=============================="
    )

    print(
        "Predicted Waste Type:",
        prediction
    )

    print(
        "=============================="
    )


    print(
        "\nRecommendation:"
    )


    print(
        ml.get_recommendation(
            prediction
        )
    )





# =====================================================
# MAIN
# =====================================================

if __name__=="__main__":

    predict()