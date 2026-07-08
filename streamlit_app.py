import streamlit as st
import pandas as pd
import joblib

# Load pipeline components
hpp = joblib.load("house_rent_pipeline.pkl")

e = hpp["encoder"]
s = hpp["scaler"]
m = hpp["model"]

st.set_page_config(page_title="House Rent Prediction", page_icon="🏠")
st.sidebar.title("🏠 House Rent Prediction")

st.sidebar.markdown("## 👩‍💻 Developer")
st.sidebar.write("Anushka Upadhyay")

st.sidebar.markdown("## 🔗 GitHub")
st.sidebar.write("https://github.com/Anushkaupa08")

st.sidebar.markdown("---")
st.sidebar.info("Predict monthly house rent using Machine Learning.")

st.title("🏠 House Rent Prediction")
st.caption("Enter house details to predict monthly rent.")

# User Inputs
col1, col2 = st.columns(2)

with col1:
    bhk = st.number_input("BHK", min_value=1, max_value=20, value=2)
    size = st.number_input("Size (sq.ft)", min_value=100, max_value=10000, value=1200)
    city = st.text_input("City", "Delhi")

with col2:
    area_type = st.selectbox("Area Type",
        ["Super Area", "Carpet Area", "Built Area"])

    furnishing = st.selectbox("Furnishing Status",
        ["Furnished", "Semi-Furnished", "Unfurnished"])

    tenant = st.selectbox("Tenant Preferred",
        ["Bachelors", "Family", "Bachelors/Family"])
    bathroom = st.number_input(
    "Bathrooms",
    min_value=1,
    max_value=20,
    value=2
)

contact = st.selectbox(
    "Point of Contact",
    ["Contact Owner", "Contact Agent", "Contact Builder"]
)
# Prediction
if st.button("Predict Rent"):

    new_house = pd.DataFrame({
        "BHK": [bhk],
        "Size": [size],
        "Area Type": [area_type],
        "City": [city],
        "Furnishing Status": [furnishing],
        "Tenant Preferred": [tenant],
        "Bathroom": [bathroom],
        "Point of Contact": [contact]
    })

    try:
        encoded = e.transform(new_house)
        scaled = s.transform(encoded)

        prediction = m.predict(scaled)[0]

        st.success(f"🏠 Predicted Monthly Rent: ₹ {prediction:,.0f}")
        st.info("💡 This prediction is based on the trained Machine Learning model.")

    except Exception as ex:
        st.error(f"Prediction Error: {ex}")
