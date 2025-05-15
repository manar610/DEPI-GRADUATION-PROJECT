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

# Left side: Input features
with left_col:
    st.header("🔧 Input Features")
    col1, col2 = st.columns(2)
    with col1:
        credit_score = st.slider("Credit Score", min_value=300, max_value=900, value=600)
        geography = st.selectbox("Geography", sorted(data["Geography"].unique()))
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.slider("Age", 18, 100, 30)
        tenure = st.slider("Tenure (Years)", 0, 10, 1)
    with col2:
        balance = st.number_input("Account Balance", min_value=0.0)
        num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
        has_card = st.selectbox("Has Credit Card", ["Yes", "No"])
        is_active = st.selectbox("Is Active Member", ["Yes", "No"])
        salary = st.number_input("Estimated Salary", min_value=0.0)

# Right side: Prediction & Visuals
with right_col:
    st.header("📈 Prediction Output")

    # Format the input for prediction
    input_dict = {
        'CreditScore': credit_score,
        'Geography': geography,
        'Gender': gender,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_products,
        'HasCrCard': 1 if has_card == "Yes" else 0,
        'IsActiveMember': 1 if is_active == "Yes" else 0,
        'EstimatedSalary': salary
    }

    df_input = pd.DataFrame([input_dict])

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
