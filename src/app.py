import streamlit as st
import pandas as pd
import joblib


# Load Models and Encoders
model = joblib.load("delay_prediction_model.pkl")
encoders = joblib.load("label_encoders.pkl")

if not isinstance(encoders, dict):
    encoders = dict(encoders)

st.set_page_config(page_title="Logistic Delay Prediction", page_icon="🚚", layout="centered")

st.title("🚚 Logistic Delay Prediction App")
st.write("Enter the delivery details below and predict whether the shipment will be delayed.")

# Input Feilds
st.subheader("📌 Enter Delivery Details")

distance = st.number_input(
    "Distance (km)",
    min_value=0.0,
    max_value=500.0,
    value=10.0,
    step=1.0,
    help="Distance between warehouse and destination."
)
traffic = st.selectbox(
    "Traffic Level",
    ["Low", "Medium", "High"],
    help="Current traffic condition."
)

weather = st.selectbox(
    "Weather Condition",
    ["Clear","Rain","Fog"],
    help="Weather during delivery."
)

# Encoder User Inputs
traffic_encoded = encoders["Traffic"].transform([traffic])[0]
weather_encoded = encoders["Weather"].transform([weather])[0]

input_data = pd.DataFrame([{
    "Distance_km": distance,
    "Traffic": traffic_encoded,
    "Weather":weather_encoded
}])

# Prediction Button
if st.button("🔍 Predict Delay"):
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1] * 100


    st.write("---")

    if prediction == 1:
        st.error(F"🚨 Delivery is likely to be **DELAYED**.\n\n**Delay Probability:**{prob:.2f}%")
    else:
        st.success(f"✅ Delivery is likely **ON TIME**.\n\n**Delay Probability:**{prob:.2f}%")

# Feature Important Section
st.write("---")
st.subheader("📊 Model Feature Importance")

try:
    importance = model.feature_importances_
    features = ["Distance_km","Traffic","Weather"]
    fi_df= pd.DataFrame({"Feature": features, "Importance": importance})
    st.bar_chart(fi_df.set_index("Feature"))
except:
    st.info("Feature importance not available for this model.")
