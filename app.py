import streamlit as st
import joblib
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Workforce Salary Analysis",
    page_icon="💼",
    layout="centered"
)

# ---------------- TITLE ----------------
st.title("📊 Workforce Salary Analysis & Prediction")
st.markdown("Analyze employee data and predict salary using machine learning")

st.divider()

# ---------------- INFO ----------------
st.info(
    "ℹ️ Enter values carefully:\n"
    "- Satisfaction level should be between 1 (low) and 10 (high)\n"
    "- Average monthly hours typically range between 150–300"
)

# ---------------- INPUT SECTION ----------------
col1, col2 = st.columns(2)

with col1:
    years_at_company = st.number_input(
        "📅 Years at Company",
        min_value=0,
        max_value=40,
        value=5,
        step=1
    )

with col2:
    satisfaction_level = st.number_input(
        "😊 Satisfaction Level (1–10)",
        min_value=1,
        max_value=10,
        value=5,
        step=1
    )

average_monthly_hours = st.number_input(
    "⏱️ Average Monthly Hours",
    min_value=50,
    max_value=400,
    value=200,
    step=10
)

st.divider()

# ---------------- INPUT VALIDATION ----------------
validation_messages = []

# Subtle unrealistic combo
if years_at_company == 0 and average_monthly_hours >= 140:
    validation_messages.append(
        "⚠️ This combination looks unusual (0 years with significant working hours)."
    )

# Very high workload
if average_monthly_hours > 320:
    validation_messages.append(
        "⚠️ Monthly hours are quite high — please verify."
    )

# Low satisfaction + high hours
if satisfaction_level <= 3 and average_monthly_hours > 250:
    validation_messages.append(
        "⚠️ Low satisfaction with high workload may indicate inconsistent input."
    )

# Show warnings (non-blocking)
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
        st.info("👉 You can still proceed, but review the warnings above.")

    X = [[years_at_company, satisfaction_level, average_monthly_hours]]

    try:
        X_scaled = scaler.transform(X)
        prediction = model.predict(X_scaled)[0]

        # ---------------- OUTPUT ----------------
        st.success(f"💰 Predicted Salary: ₹ {prediction:,.2f}")
        st.metric("Estimated Salary", f"₹ {prediction:,.0f}")

    except Exception as e:
        st.error(f"❌ Prediction error: {e}")

else:
    st.warning("⚠️ Enter inputs and click 'Predict Salary'")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit | HR Analytics Project")
