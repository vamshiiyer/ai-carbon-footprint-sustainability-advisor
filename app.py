import streamlit as st
import pickle

st.set_page_config(page_title="AI Sustainability Advisor", layout="centered")

# Load trained model
with open("carbon_model.pkl", "rb") as f:
    model, encoder = pickle.load(f)

st.title("🌱 AI-Based Carbon Footprint & Sustainability Advisor")
st.write("Estimate your environmental impact and get eco-friendly suggestions.")

# User inputs
transport = st.number_input("🚗 Daily transport distance (km)", min_value=0, max_value=100, value=10)
electricity = st.number_input("⚡ Monthly electricity usage (units)", min_value=0, max_value=500, value=120)
diet = st.selectbox("🍽️ Diet type", ["veg", "mixed", "non-veg"])
household = st.slider("🏠 Household size", min_value=1, max_value=10, value=3)

# Encode diet
diet_encoded = encoder.transform([diet])[0]

if st.button("Analyze Sustainability Impact"):
    prediction = model.predict([[transport, electricity, diet_encoded, household]])[0]

    st.subheader(f"🌍 Impact Level: {prediction}")

    if prediction == "High":
        st.error("High carbon footprint detected. Consider reducing vehicle use and saving electricity.")
    elif prediction == "Medium":
        st.warning("Moderate footprint. You can switch to energy-efficient appliances.")
    else:
        st.success("Low footprint. Great job maintaining a sustainable lifestyle!")

st.caption("⚠️ Educational project for sustainability awareness.")
