import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier

st.set_page_config(layout="wide")

# Load model
model = CatBoostClassifier()
model_path = os.path.join(os.path.dirname(__file__), "catboost_model.cbm")
model.load_model(model_path)

# Load dataset for visualizations
data_path = os.path.join(os.path.dirname(__file__), "final_cleaned_data.csv")
data = pd.read_csv(data_path)

st.title("Customer Churn Prediction")

left_col, right_col = st.columns([1, 2])
with left_col:
    st.header("🔧 Input Features")
    col1, col2 = st.columns(2)
    with col1:

    credit_score = st.slider("Credit Score", 300, 850, 600)
    geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age", 18, 100, 30)
    tenure = st.slider("Tenure (Years)", 0, 10, 3)
    with col2:
    balance = st.number_input("Balance", min_value=0.0, value=0.0)
    num_of_products = st.slider("Number of Products", 1, 4, 1)
    has_cr_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
    is_active_member = st.selectbox("Is Active Member?", ["Yes", "No"])
    estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)

# Encode inputs to match training format
input_data = {
    "CreditScore": credit_score,
    "Geography": {"France": 0, "Germany": 1, "Spain": 2}[geography],
    "Gender": 1 if gender == "Male" else 0,
    "Age": age,
    "Tenure": tenure,
    "Balance": balance,
    "NumOfProducts": num_of_products,
    "HasCrCard": 1 if has_cr_card == "Yes" else 0,
    "IsActiveMember": 1 if is_active_member == "Yes" else 0,
    "EstimatedSalary": estimated_salary
}

df_input = pd.DataFrame([input_data])

with right_col:
    st.header("📈 Prediction Output")

    if st.button("Predict Churn"):
        prediction = model.predict(df_input)[0]
        prob = model.predict_proba(df_input)[0][1]
        
        if prediction == 1:
            st.error(f"⚠️ Likely to Churn (Confidence: {prob:.2%})")
        else:
            st.success(f"✅ Unlikely to Churn (Confidence: {1 - prob:.2%})")

    # Monthly Charges KDE (alternative: Salary)
    st.subheader("📊 Churned vs. Retained - Estimated Salary")
    fig_kde, ax_kde = plt.subplots()
    sns.kdeplot(data[data['Exited'] == 1]['EstimatedSalary'], label="Churned", shade=True, color="#ff7f0e", ax=ax_kde)
    sns.kdeplot(data[data['Exited'] == 0]['EstimatedSalary'], label="Retained", shade=True, color="#1f77b4", ax=ax_kde)
    ax_kde.set_xlabel("Estimated Salary")
    ax_kde.set_ylabel("Density")
    ax_kde.legend()

    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(fig_kde)

    # Pie chart of churn rate
    churn_counts = data['Exited'].value_counts()
    labels = ['Retained', 'Churned']
    colors = ['#1f77b4', '#ff7f0e']
    fig_pie, ax_pie = plt.subplots()
    ax_pie.pie(churn_counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    ax_pie.axis('equal')
    with col2:
        st.pyplot(fig_pie)
