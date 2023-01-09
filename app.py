import streamlit as st
import pandas as pd
import joblib

# Title
st.header("Heart Disease Prediction using Bayesian Belief Networks")

# Inputs

age = st.number_input("Age ", min_value=0, max_value=100, value=41, step=1)
sex = st.number_input("Sex ", min_value=0, max_value=1, value=0, step=1)
cp = st.number_input("Chest pain type (0: typical angina 1: atypical angina 2: non-anginal pain 3: asymptomatic)", min_value=0, max_value=3, value=1, step=1)
trestbps = st.number_input("Resting blood pressure", min_value=0, max_value=250, value=130, step=1)
chol = st.number_input("Serum cholestoral in mg/dl", min_value=0, max_value=500, value=204, step=1)
fbs = st.number_input("Fasting blood sugar > 120 mg/dl (1: true; 0: false)", min_value=0, max_value=1, value=0, step=1)
thalach = st.number_input("Maximum heart rate achieved", min_value=0, max_value=500, value=172, step=1)
exang = st.number_input("Exercise induced angina (1: yes; 0: no)", min_value=0, max_value=1, value=0, step=1)
# cholesterol = st.slider("Cholesterol (mg/dL)", min_value=100, max_value=300, value=200)
# blood_pressure = st.slider("Blood Pressure (mm Hg)", min_value=80, max_value=220, value=120)

# If button is pressed
if st.button("Submit"):

    # Unpickel model
    model = joblib.load("model.pkl")

    # Storing the unputs into a dataframe

    X = pd.DataFrame([['age', 'sex', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach','exang']], columns=['age', 'sex', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach','exang'])

    # Getting the prediction
    prediction = model.predict(X)[0]

    # Printing the prediction
    st.text(f"This prediction is a {prediction}")