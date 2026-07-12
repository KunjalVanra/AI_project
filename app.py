import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

import waste_management_ml as ml


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(

    page_title="AI Smart Waste Classification",

    page_icon="♻️",

    layout="wide"

)


# =====================================================
# LOAD FILES
# =====================================================

@st.cache_resource
def load_files():

    model = joblib.load(
        "waste_classifier.pkl"
    )


    encoders = joblib.load(
        "encoders.pkl"
    )


    target_encoder = joblib.load(
        "target_encoder.pkl"
    )


    return model, encoders, target_encoder



# =====================================================
# PREPROCESS INPUT
# =====================================================

def preprocess_input(data, encoders):


    data = data.copy()


    for col, encoder in encoders.items():


        data[col] = encoder.transform(
            data[col]
        )



    data["Recyclable"] = data["Recyclable"].map(
        {
            "Yes":1,
            "No":0
        }
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


    return data



# =====================================================
# HEADER
# =====================================================


st.title(
    "♻️ AI-Based Smart Waste Classification System"
)


st.markdown(
"""
### Machine Learning Based Waste Segregation System

This application predicts waste categories using a
Random Forest Machine Learning model and provides
appropriate disposal recommendations.
"""
)



# =====================================================
# LOAD MODEL
# =====================================================

try:

    model, encoders, target_encoder = load_files()


except Exception as e:


    st.error(
        "Model files missing. Run main.py first."
    )

    st.stop()



# =====================================================
# SIDEBAR INPUT
# =====================================================


st.sidebar.header(
    "Waste Information"
)


material = st.sidebar.selectbox(

    "Material",

    encoders["Material"].classes_

)


toxicity = st.sidebar.selectbox(

    "Toxicity",

    encoders["Toxicity"].classes_

)


recyclable = st.sidebar.selectbox(

    "Recyclable",

    [

        "Yes",
        "No"

    ]

)


energy = st.sidebar.selectbox(

    "Energy Recovery",

    encoders["Energy_Recovery"].classes_

)


city = st.sidebar.selectbox(

    "City",

    encoders["City"].classes_

)



# =====================================================
# PREDICTION
# =====================================================


if st.button(
    "🔍 Predict Waste Type"
):


    input_data = pd.DataFrame(

        [{

        "Material":material,

        "Toxicity":toxicity,

        "Recyclable":recyclable,

        "Energy_Recovery":energy,

        "City":city

        }]

    )


    processed = preprocess_input(

        input_data,

        encoders

    )


    prediction = model.predict(
        processed
    )[0]


    probability = model.predict_proba(
        processed
    )[0]



    # Convert number back to label

    prediction = target_encoder.inverse_transform(
        [prediction]
    )[0]



    confidence = max(probability)*100



    st.success(
        f"### Predicted Waste Type: {prediction}"
    )


    st.metric(

        "Confidence",

        f"{confidence:.2f}%"

    )



    st.subheader(
        "Recommended Disposal Method"
    )


    st.info(

        ml.get_recommendation(
            prediction
        )

    )



    # ------------------------------
    # Probability Chart
    # ------------------------------


    st.subheader(
        "Prediction Probability"
    )


    labels = target_encoder.classes_


    prob_df = pd.DataFrame(

        {

        "Waste Type":labels,

        "Probability":probability

        }

    )


    st.bar_chart(

        prob_df.set_index(
            "Waste Type"
        )

    )



# =====================================================
# MODEL DETAILS
# =====================================================


st.divider()


st.header(
    "📊 Model Information"
)



col1,col2,col3 = st.columns(3)


with col1:

    st.metric(
        "Algorithm",
        "Random Forest"
    )


with col2:

    st.metric(
        "Features",
        "5"
    )


with col3:

    st.metric(
        "Dataset",
        "10000 Records"
    )



# =====================================================
# SHOW GENERATED GRAPHS
# =====================================================


st.header(
    "Model Analysis"
)


try:


    st.image(

        "confusion_matrix.png",

        caption="Confusion Matrix"

    )


    st.image(

        "feature_importance.png",

        caption="Feature Importance"

    )


except:

    st.warning(
        "Train model to generate graphs."
    )



# =====================================================
# ABOUT
# =====================================================


st.divider()


st.subheader(
    "Technology Used"
)


st.write(
"""
✔ Python  
✔ Pandas  
✔ Scikit-Learn  
✔ Random Forest Algorithm  
✔ Artificial Intelligence  
✔ Streamlit Dashboard  

### Project Features

✔ Automatic Waste Classification  
✔ Machine Learning Prediction  
✔ Confidence Score  
✔ Smart Disposal Recommendation  
✔ Data Visualization  
"""
)