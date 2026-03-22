import streamlit as st
import joblib
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Salary Predictor",
    page_icon="💼",
    layout="centered"
)

# ---------------- TITLE ----------------
st.title("💼 Employee Salary Prediction App")
st.markdown("### Predict employee salary using Machine Learning")

st.divider()

# ---------------- INFO ----------------
st.info(
    "ℹ️ Provide employee details below. "
    "Satisfaction level ranges from 1 (low) to 10 (high)."
)

# ---------------- INPUT SECTION ----------------
col1, col2 = st.columns(2)

with col1:
    years_at_company = st.slider("📅 Years at Company", 0, 20, 5)

with col2:
    satisfaction_level = st.slider("😊 Satisfaction Level (1–10)", 1, 10, 5)

average_monthly_hours = st.slider("⏱️ Average Monthly Hours", 120, 400, 200)

st.divider()

# ---------------- INPUT VALIDATION ----------------
validation_messages = []

# Rule 1: Unrealistic combo
if years_at_company == 0 and average_monthly_hours > 250:
    validation_messages.append(
        "⚠️ High working hours with 0 years at company seems unrealistic."
    )

# Rule 2: Extremely high workload
if average_monthly_hours > 350:
    validation_messages.append(
        "⚠️ Monthly hours seem very high. Please verify."
    )

# Rule 3: Low satisfaction + high hours
if satisfaction_level <= 3 and average_monthly_hours > 250:
    validation_messages.append(
        "⚠️ Low satisfaction with high workload may indicate unusual data."
    )

# Show warnings
for msg in validation_messages:
    st.warning(msg)

# ---------------- LOAD MODEL ----------------
try:
    scaler = joblib.load("scaler.pkl")
    model = joblib.load("model.pkl")
except Exception as e:
    st.error(f"❌ Error loading model files: {e}")
    st.stop()

# ---------------- PREDICTION ----------------
if st.button("🔍 Predict Salary"):

    if len(validation_messages) > 0:
        st.info("👉 You can still proceed, but please review the warnings above.")

    # Prepare input
    X = [[years_at_company, satisfaction_level, average_monthly_hours]]

    try:
        X_scaled = scaler.transform(X)
        prediction = model.predict(X_scaled)[0]

        # ---------------- OUTPUT ----------------
        st.success(f"💰 Predicted Salary: ₹ {prediction:,.2f}")

        # Optional metric display (nice UI touch)
        st.metric(label="Estimated Salary", value=f"₹ {prediction:,.0f}")

    except Exception as e:
        st.error(f"❌ Prediction error: {e}")

else:
    st.warning("⚠️ Adjust inputs and click 'Predict Salary'")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit | Machine Learning Project for HR Analytics")
