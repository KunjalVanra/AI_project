# AI_project
# ♻️ AI-Based Smart Waste Classification System

## 📌 Project Overview

The **AI-Based Smart Waste Classification System** is a Machine Learning application developed to automatically classify different types of waste and provide suitable disposal recommendations.

The system uses waste-related parameters such as material type, toxicity level, recyclability, energy recovery capability, and city information to predict the waste category.

This project aims to support automated waste segregation and improve smart waste management practices using Artificial Intelligence.

---

# 🎯 Objectives

- Develop an AI-based waste classification system.
- Automatically categorize waste into different classes.
- Reduce manual waste segregation effort.
- Provide disposal recommendations based on waste type.
- Analyze model performance using machine learning evaluation metrics.

---

# 🚀 Features

## Machine Learning Features

✅ Waste Type Classification  
✅ Random Forest Based Prediction  
✅ Model Accuracy Evaluation  
✅ Precision, Recall and F1-score Calculation  
✅ Confusion Matrix Visualization  
✅ Feature Importance Analysis  


## Application Features

✅ User-friendly Streamlit Interface  
✅ Manual Waste Information Input  
✅ Prediction Confidence Score  
✅ Disposal Recommendation System  
✅ Data Visualization Dashboard  

---

# 🏗️ System Architecture
            Waste Dataset
                 |
                 ↓
        Data Preprocessing
                 |
                 ↓
         Feature Encoding
                 |
                 ↓
      Random Forest Classifier
                 |
                 ↓
        Waste Type Prediction
                 |
                 ↓
      Disposal Recommendation
                 |
                 ↓
         Streamlit Dashboard

        
---

# 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Programming Language |
| Pandas | Data Processing |
| NumPy | Numerical Operations |
| Scikit-Learn | Machine Learning |
| Random Forest | Classification Algorithm |
| Matplotlib | Data Visualization |
| Seaborn | Graph Generation |
| Streamlit | Web Application Interface |
| Joblib | Model Saving and Loading |

---

# 📂 Project Structure
AI-Waste-Classification/

│
├── main.py
│ └── Main application and prediction system
│
├── waste_management_ml.py
│ └── Machine Learning model training
│
├── waste_management_dl.py
│ └── Deep Learning module
│
├── app.py
│ └── Streamlit web application
│
├── Waste_Dataset_With_City_10000.csv
│ └── Dataset file
│
├── waste_classifier.pkl
│ └── Trained ML model
│
├── target_encoder.pkl
│ └── Output label encoder
│
├── encoders.pkl
│ └── Feature encoders
│
├── confusion_matrix.png
│ └── Model evaluation graph
│
├── feature_importance.png
│ └── Feature analysis graph
│
└── README.md



---

# 📊 Dataset Description

The dataset contains waste information with the following attributes:

| Feature | Description |
|---------|-------------|
| Waste_Type | Category of waste |
| Material | Material composition |
| Toxicity | Toxicity level |
| Recyclable | Recycling possibility |
| Energy_Recovery | Energy recovery potential |
| City | Waste location |

---

# 🗂️ Waste Categories

The model classifies waste into:

- E-Waste
- Plastic
- Paper
- Metal
- Organic

---

# ⚙️ Installation

## 1. Clone or Download Project

Download the project files into your system.

---

## 2. Create Virtual Environment

```bash
python -m venv myenv

