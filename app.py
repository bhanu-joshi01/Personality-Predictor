import streamlit as st
import joblib
import numpy as np

# Load model, scaler, and label encoder
model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")
label_encoder = joblib.load("models/labelEncode.pkl")

# Page config
st.set_page_config(page_title="Personality Predictor", layout="centered")

# Title and author credit
st.title("🧠 Personality Predictor App")
st.markdown("#### Made by Bhanu Joshi❤️")

# Input fields
time_spent = st.slider("Time Spent Alone", 0, 15, 5)
social_attendance = st.slider("Social Event Attendance", 0, 15, 5)
going_outside = st.slider("Frequency of Going Outside", 0, 15, 5)
friend_circle = st.slider("Friends Circle Size", 0, 15, 5)
post_freq = st.slider("Social Media Post Frequency", 0, 15, 5)

drained = st.selectbox("Drained After Socializing", ["Yes", "No"])
stage_fear = st.selectbox("Stage Fear", ["Yes", "No"])

# Convert categorical inputs
drained_val = 1 if drained == "Yes" else 0
stage_fear_val = 1 if stage_fear == "Yes" else 0

# Prediction
if st.button("Predict Personality"):
    input_data = np.array([[time_spent, social_attendance, going_outside,
                            friend_circle, post_freq, drained_val, stage_fear_val]])
    scaled = scaler.transform(input_data)
    pred = model.predict(scaled)
    result = label_encoder.inverse_transform(pred)[0]

    # Show result with color-coded box
    if result.lower() == "extrovert":
        st.success(f"🧬 Predicted Personality: **{result}**")
    else:
        st.warning(f"🧬 Predicted Personality: **{result}**")

# Footer
st.markdown("---")
st.markdown(
    """
    💡 **Disclaimer:**  
    This is a machine learning-based prediction for educational and experimental purposes only.  
    For professional advice or accurate diagnosis, always consult a certified mental health expert.

    🔗 *Made with ❤️ by Bhanu Joshi*
    """
)




