# ============================================================
# Walmart Sales Prediction – Streamlit App (FINAL VERSION)
# Loads MLflow model from Databricks using Streamlit Secrets
# Dark UI • AI Insights • Feature Importance • Forecast
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow.pyfunc
import base64
import os
from openai import OpenAI

# ============================================================
# 1. LOAD DATABRICKS CREDENTIALS FROM STREAMLIT SECRETS
# ============================================================
os.environ["DATABRICKS_HOST"] = st.secrets["DATABRICKS_HOST"]
os.environ["DATABRICKS_TOKEN"] = st.secrets["DATABRICKS_TOKEN"]

# MLflow tracking to Databricks
mlflow.set_tracking_uri("databricks")

# Your registered model path
MODEL_URI = "models:/workspace.jzhao221.walmart/1"

# ============================================================
# 2. LOAD MODEL FROM DATABRICKS
# ============================================================
@st.cache_resource
def load_model():
    return mlflow.pyfunc.load_model(MODEL_URI)

model = load_model()

# ============================================================
# 3. LOAD BACKGROUND + LOGO
# ============================================================
def get_base64(path):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except:
        return None

bg64 = get_base64("background.jpg")
logo64 = get_base64("logo.png")

# ============================================================
# 4. OPENAI CLIENT (KEY IN SECRETS)
# ============================================================
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def ai_insight(title, explanation, values):
    prompt = f"""
    Explain this chart in simple business English.
    Avoid ML terminology.

    Title: {title}
    Explanation: {explanation}
    Values: {values}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200
        )
        return response.choices[0].message.content.strip()
    except:
        return "(AI insight unavailable)"

# ============================================================
# 5. PAGE DESIGN (DARK THEME + GLASS CARD)
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
        padding: 30px;
        border-radius: 15px;
        color: white;
        backdrop-filter: blur(8px);
        box-shadow: 0 4px 18px rgba(0,0,0,0.5);
    }}
    .pred-box {{
        background: #0EA5E9;
        padding: 18px;
        border-radius: 14px;
        font-size: 26px;
        text-align: center;
        color: white;
        margin-top: 10px;
    }}
    </style>
    """, unsafe_allow_html=True)


# ============================================================
# 6. TITLE
# ============================================================
st.markdown("<h1 style='text-align:center;color:#38BDF8;'>Walmart Weekly Sales Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#7dd3fc;'>Dark Theme • Sky Blue Accents • AI Insights</p>", unsafe_allow_html=True)

# ============================================================
# 7. INPUT PANEL
# ============================================================
st.markdown('<div class="main-card">', unsafe_allow_html=True)

st.header("🔧 Enter Store Information")

store = st.number_input("Store ID", 1, 50, 1)
dept = st.number_input("Dept ID", 1, 99, 1)
holiday = st.selectbox("Holiday Flag", [0, 1])
temp = st.number_input("Temperature (°F)", value=70.0)
fuel = st.number_input("Fuel Price ($)", value=2.50)
cpi = st.number_input("CPI", value=220.0)
unemp = st.number_input("Unemployment (%)", value=5.0)
year = st.number_input("Year", value=2023)
month = st.number_input("Month", 1, 12, 1)
week = st.number_input("Week", 1, 53, 1)

df = pd.DataFrame({
    "Store": [store],
    "Dept": [dept],
    "Holiday_Flag": [holiday],
    "Temperature": [temp],
    "Fuel_Price": [fuel],
    "CPI": [cpi],
    "Unemployment": [unemp],
    "Year": [year],
    "Month": [month],
    "Week": [week]
}).astype(float)

# ============================================================
# 8. PREDICTION
# ============================================================
if st.button("Predict Weekly Sales"):
    pred = float(model.predict(df)[0])
    st.markdown(f'<div class="pred-box"><b>${pred:,.2f}</b></div>', unsafe_allow_html=True)

# ============================================================
# 9. FEATURE IMPORTANCE (Sensitivity)
# ============================================================
st.subheader("📌 Feature Importance")

base = float(model.predict(df)[0])
importance = {}

for col in df.columns:
    t = df.copy()
    t[col] *= 1.10
    importance[col] = abs(float(model.predict(t)[0]) - base)

importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

fig, ax = plt.subplots()
ax.barh(list(importance.keys()), list(importance.values()), color="#38BDF8")
ax.invert_yaxis()
st.pyplot(fig)

st.write(ai_insight("Feature Importance", "Drivers of weekly sales.", importance))

# ============================================================
# 10. 10-WEEK FORECAST
# ============================================================
st.subheader("📈 10-Week Forecast")

future_weeks = np.arange(week, week + 10)
df_future = df.loc[df.index.repeat(10)].copy()
df_future["Week"] = future_weeks

preds = model.predict(df_future)

fig2, ax2 = plt.subplots()
ax2.plot(future_weeks, preds, marker="o", color="#7dd3fc")
st.pyplot(fig2)

st.write(ai_insight("10-Week Forecast", "Predicted sales trend.", {
    "weeks": list(future_weeks),
    "sales": list(preds)
}))

st.markdown("</div>", unsafe_allow_html=True)
