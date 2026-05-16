import streamlit as st
import pandas as pd
import pickle

model_data = pickle.load(open("churn_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

model = model_data["model"]
feature_cols = model_data["feature_cols"]
le_encoders = model_data["le_encoders"]
numerical_cols = model_data["numerical_cols"]
binary_cols = model_data["binary_cols"]
multi_cols = model_data["multi_cols"]

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📡",
    layout="centered"
)

st.title("📡 Customer Churn Prediction System")
st.subheader("Enter Customer Details")

col1, col2, col3 = st.columns([1,1,1], gap="small")

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])

with col2:
    internet = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    security = st.selectbox(
        "Online Security",
        ["No", "Yes", "No internet service"]
    )

    backup = st.selectbox(
        "Online Backup",
        ["No", "Yes", "No internet service"]
    )

    protection = st.selectbox(
        "Device Protection",
        ["No", "Yes", "No internet service"]
    )

    support = st.selectbox(
        "Tech Support",
        ["No", "Yes", "No internet service"]
    )

with col3:
    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    payment = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    tenure = st.slider("Tenure", 0, 72, 12)

    monthly = st.number_input(
        "Monthly Charges",
        0.0,
        200.0,
        70.0
    )

    total = st.number_input(
        "Total Charges",
        0.0,
        10000.0,
        1000.0
    )

if st.button("Predict Churn"):

    input_data = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": protection,
        "TechSupport": support,
        "Contract": contract,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }])

    for col in binary_cols:
        if col in input_data.columns:
            le = le_encoders[col]
            input_data[col] = le.transform(input_data[col])

    ohe_cols = [
        c for c in multi_cols
        if c in input_data.columns
    ]

    input_data = pd.get_dummies(
        input_data,
        columns=ohe_cols,
        drop_first=True
    )

    input_data = input_data.reindex(
        columns=feature_cols,
        fill_value=0
    )

    input_data[numerical_cols] = scaler.transform(
        input_data[numerical_cols]
    )

    probability = model.predict_proba(input_data)[0][1]

    prediction = 1 if probability >= 0.18 else 0

    st.subheader("Prediction Result")

    st.write(f"### Churn Probability : {probability:.2%}")

    st.write(f"### Churn Value : {prediction}")

    if prediction == 1:
        st.error("⚠️ Customer is Likely to Churn")
    else:
        st.success("✅ Customer is Likely to Stay")