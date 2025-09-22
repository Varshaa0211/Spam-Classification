import streamlit as st
import joblib
import re

# ----------------------------
# Load Model & Vectorizer
# ----------------------------
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="📩 Spam Classifier App",
    page_icon="🚀",
    layout="centered"
)

# ----------------------------
# Custom CSS for styling
# ----------------------------
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(120deg, #d4fc79, #96e6a1);
        font-family: 'Trebuchet MS', sans-serif;
    }
    .title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        color: #2c3e50;
    }
    .result-box {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        color: white;
    }
    .spam {
        background-color: #e74c3c;
    }
    .ham {
        background-color: #27ae60;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------
# Title & Description
# ----------------------------
st.markdown("<h1 class='title'>📩 Spam Classification App 🚀</h1>", unsafe_allow_html=True)
st.write("🔍 Enter a message below and find out if it's **Spam 🚫** or **Ham ✅**")

# ----------------------------
# User Input
# ----------------------------
user_input = st.text_area("✍️ Enter your message here:")

if st.button("🔮 Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a message to check!")
    else:
        # Preprocess & predict
        input_data = vectorizer.transform([user_input])
        prediction = model.predict(input_data)[0]

        if prediction == 1:  # Spam
            st.markdown("<div class='result-box spam'>🚫 This Message is SPAM! 🚫</div>", unsafe_allow_html=True)
        else:  # Ham
            st.markdown("<div class='result-box ham'>✅ This Message is NOT Spam (HAM)! ✅</div>", unsafe_allow_html=True)

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.markdown("🌟 Developed by Varsha❤️")
