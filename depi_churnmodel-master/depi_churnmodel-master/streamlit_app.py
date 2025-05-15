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
    col1, col2 = st.columns([1, 1])
    with col1:
        credit_score = st.slider("Credit Score", 300, 900, 600)
        geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.slider("Age", 18, 100, 30)
        tenure = st.slider("Tenure (Years)", 0, 10, 3)
    with col2:
        balance = st.number_input("Balance", min_value=0.0)
        num_products = st.slider("Number of Products", 1, 4, 1)
        has_cr_card = st.selectbox("Has Credit Card", ["Yes", "No"])
        is_active = st.selectbox("Is Active Member", ["Yes", "No"])
        salary = st.number_input("Estimated Salary", min_value=0.0)

trained_feature_order = [
    'CreditScore', 'Age', 'Tenure', 'Balance','NumOfProducts', 
    'HasCrCard', 'IsActiveMember', 'EstimatedSalary','Geography_France', 'Geography_Germany', 'Geography_Spain',
    'Gender_Female', 'Gender_Male'
]

# Create input dict with encoded values
input_data = {
    'CreditScore': credit_score,
    'Geography_France': 1 if geography == "France" else 0,
    'Geography_Germany': 1 if geography == "Germany" else 0,
    'Geography_Spain': 1 if geography == "Spain" else 0,
    'Gender_Female': 1 if gender == "Female" else 0,
    'Gender_Male': 1 if gender == "Male" else 0,
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': num_products,
    'HasCrCard': 1 if has_cr_card == "Yes" else 0,
    'IsActiveMember': 1 if is_active == "Yes" else 0,
    'EstimatedSalary': salary
}

# Create DataFrame and reorder columns
df_input = pd.DataFrame([input_data])
df_input = df_input.reindex(columns=trained_feature_order)
with right_col:
    st.header("📈 Prediction Output")

# Predict
    if st.button("Predict Churn"):
        prediction = model.predict(df_input)[0]
        prob = model.predict_proba(df_input)[0][1]
    
        if prediction == 1:
            st.error(f"⚠️ Likely to Churn (Confidence: {prob:.2%})")
        else:
            st.success(f"✅ Unlikely to Churn (Confidence: {1 - prob:.2%})")

    st.subheader("📊 Churned vs. Retained - Age")
    fig_kde_age, ax_kde_age = plt.subplots()
    sns.kdeplot(data[data['Exited'] == 1]['Age'], label="Churned", fill=True, color="#ff7f0e", ax=ax_kde_age)
    sns.kdeplot(data[data['Exited'] == 0]['Age'], label="Retained", fill=True, color="#1f77b4", ax=ax_kde_age)
    ax_kde_age.set_xlabel("Age")
    ax_kde_age.set_ylabel("Density")
    ax_kde_age.legend()
    

    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(fig_kde_age)

    # Pie chart of churn rate
    churn_counts = data['Exited'].value_counts()
    labels = ['Retained', 'Churned']
    colors = ['#1f77b4', '#ff7f0e']
    fig_pie, ax_pie = plt.subplots()
    ax_pie.pie(churn_counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    ax_pie.axis('equal')
    with col2:
        st.pyplot(fig_pie)
