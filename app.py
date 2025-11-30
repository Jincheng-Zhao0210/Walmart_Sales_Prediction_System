# ============================================================
# Walmart Weekly Sales Prediction App
# Includes: OpenAI insights + gzip model + forecast + importance
# Works on Streamlit Cloud
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import gzip
import pickle
import base64
import matplotlib.pyplot as plt
from openai import OpenAI

# ============================================================
# 1) LOAD COMPRESSED MODEL (model.pkl.gz)
# ============================================================

MODEL_PATH = "model.pkl.gz"   # Must match your GitHub file

@st.cache_resource
def load_model():
    with gzip.open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

model = load_model()


# ============================================================
# 2) LOAD BACKGROUND + LOGO
# ============================================================

def load_image(path):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except:
        return None

bg64 = load_image("background.jpg")
logo64 = load_image("logo.png")


# ============================================================
# 3) OPENAI CLIENT (Key stored in Streamlit Secrets)
# ============================================================

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def ai_insight(title, explanation, values):
    prompt = f"""
    Explain the following chart or prediction in simple business English.
    Avoid machine learning terminology.

    Title: {title}
    Explanation: {explanation}
    Values: {values}

    Provide insights as if you are advising a Walmart regional manager.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"(AI insight unavailable: {e})"


# ============================================================
# 4) PAGE STYLE
# ============================================================

if bg64:
    st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bg64}");
        background-size: cover;
        background-attachment: fixed;
    }}
    .main-card {{
        background: rgba(0,0,0,0.70);
        padding: 28px;
        border-radius: 15px;
        color: white;
        backdrop-filter: blur(10px);
    }}
    .pred-box {{
        background: #0EA5E9;
        padding: 18px;
        border-radius: 12px;
        text-align: center;
        font-size: 24px;
        color: white;
        margin-top: 15px;
    }}
    </style>
    """, unsafe_allow_html=True)


# ============================================================
# 5) HEADER
# ============================================================

st.markdown("<h1 style='text-align:center;color:#38BDF8;'>Walmart Weekly Sales Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#93c5fd;'>AI-Enhanced Forecasting Dashboard</p>", unsafe_allow_html=True)

if logo64:
    st.markdown(
        f"<div style='text-align:center;'><img src='data:image/png;base64,{logo64}' width='160'></div>",
        unsafe_allow_html=True
    )


# ============================================================
# 6) INPUT FORM
# ============================================================

st.markdown("<div class='main-card'>", unsafe_allow_html=True)
st.header("🔧 Input Store Information")

inputs = {
    "Store": st.number_input("Store ID", 1, 50, 1),
    "Dept": st.number_input("Dept ID", 1, 99, 1),
    "Holiday_Flag": st.selectbox("Holiday Flag (0 = No, 1 = Yes)", [0, 1]),
    "Temperature": st.number_input("Temperature (°F)", value=70.0),
    "Fuel_Price": st.number_input("Fuel Price ($)", value=2.50),
    "CPI": st.number_input("CPI", value=220.0),
    "Unemployment": st.number_input("Unemployment (%)", value=5.0),
    "Year": st.number_input("Year", value=2023),
    "Month": st.number_input("Month", 1, 12, 1),
    "Week": st.number_input("Week", 1, 53, 1)
}

df = pd.DataFrame([inputs])


# ============================================================
# 7) PREDICTION
# ============================================================

st.header("📌 Predicted Weekly Sales")

if st.button("Predict Sales"):
    pred = float(model.predict(df)[0])
    st.markdown(f"<div class='pred-box'>${pred:,.2f}</div>", unsafe_allow_html=True)

    st.write("### 🧠 AI Insight")
    st.info(ai_insight(
        "Weekly Sales Prediction",
        "Interpret the predicted value in a business context.",
        {"prediction": pred}
    ))


# ============================================================
# 8) FEATURE IMPORTANCE (Sensitivity)
# ============================================================

st.header("📍 Feature Importance (Sensitivity Test)")

base = float(model.predict(df)[0])
importance = {}

for col in df.columns:
    temp = df.copy()
    temp[col] *= 1.10  # +10% change
    importance[col] = abs(float(model.predict(temp)[0]) - base)

importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

fig, ax = plt.subplots()
ax.barh(list(importance.keys()), list(importance.values()), color="#38BDF8")
ax.set_title("Feature Sensitivity")
ax.set_xlabel("Impact on Weekly Sales")
ax.invert_yaxis()
st.pyplot(fig)

st.write("### 🧠 AI Insight")
st.info(ai_insight(
    "Feature Importance",
    "Which features drive the sales prediction?",
    importance
))


# ============================================================
# 9) 10-WEEK FORECAST
# ============================================================

st.header("📈 10-Week Forecast Projection")

future_weeks = np.arange(inputs["Week"], inputs["Week"] + 10)
df_future = df.loc[df.index.repeat(10)].copy()
df_future["Week"] = future_weeks

future_preds = model.predict(df_future)

fig2, ax2 = plt.subplots()
ax2.plot(future_weeks, future_preds, marker="o", color="#00D5FF")
ax2.set_title("10-Week Sales Forecast")
ax2.set_xlabel("Week Number")
ax2.set_ylabel("Predicted Sales")
st.pyplot(fig2)

st.write("### 🧠 AI Insight")
st.info(ai_insight(
    "10-Week Forecast",
    "Explain the short-term forecast for store planning.",
    {"weeks": list(future_weeks), "sales": list(future_preds)}
))

st.markdown("</div>", unsafe_allow_html=True)
