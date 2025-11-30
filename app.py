# ============================================================
# Walmart Sales Prediction – FINAL STREAMLIT VERSION
# Databricks MLflow Model + OpenAI Insights + Dark Theme
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import mlflow.pyfunc
import matplotlib.pyplot as plt
import base64
import os
from openai import OpenAI

# ============================================================
# 1. DATABRICKS AUTH (Streamlit Secrets)
# ============================================================

# These must match your Streamlit secrets
os.environ["DATABRICKS_HOST"] = st.secrets["DATABRICKS_HOST"]
os.environ["DATABRICKS_TOKEN"] = st.secrets["DATABRICKS_TOKEN"]

# ============================================================
# 2. APP PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Walmart Sales Prediction",
    layout="wide",
)

# ============================================================
# 3. OPTIONAL BACKGROUND IMAGE
# ============================================================

def get_base64(path):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except:
        return None

bg64 = get_base64("background.jpg")  # optional

if bg64:
    st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{bg64}");
        background-size: cover;
        background-attachment: fixed;
    }}
    .main-card {{
        background: rgba(0,0,0,0.75);
        padding: 32px;
        border-radius: 18px;
        margin-top: 25px;
        color: white !important;
        backdrop-filter: blur(10px);
    }}
    .pred-box {{
        background: #0EA5E9;
        padding: 20px;
        border-radius: 14px;
        text-align: center;
        color: white;
        font-size: 26px;
    }}
    </style>
    """, unsafe_allow_html=True)

# ============================================================
# 4. OPENAI CLIENT  (Secret stored in Streamlit secrets)
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
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200
        )
        return r.choices[0].message.content.strip()
    except:
        return "(AI insight unavailable)"

# ============================================================
# 5. LOAD MODEL FROM DATABRICKS
# ============================================================

MODEL_URI = "models:/workspace.jzhao221.walmart/1"

@st.cache_resource
def load_model():
    return mlflow.pyfunc.load_model(MODEL_URI)

model = load_model()

# ============================================================
# 6. HEADER
# ============================================================

st.markdown("<h1 style='text-align:center;color:#38BDF8;'>Walmart Weekly Sales Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#7dd3fc;'>Databricks Model • AI Insights • Forecast Dashboard</p>", unsafe_allow_html=True)

st.markdown('<div class="main-card">', unsafe_allow_html=True)

# ============================================================
# 7. USER INPUTS — NO "Dept" (your model signature removed it)
# ============================================================

st.header("🔧 Enter Store Information")

store = st.number_input("Store ID", 1, 50, 1)
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
    "Holiday_Flag": [holiday],
    "Temperature": [temp],
    "Fuel_Price": [fuel],
    "CPI": [cpi],
    "Unemployment": [unemp],
    "Year": [year],
    "Month": [month],
    "Week": [week],
}).astype(float)

# ============================================================
# 8. PREDICTION
# ============================================================

st.subheader("📌 Predicted Weekly Sales")

if st.button("Predict Weekly Sales"):
    pred = float(model.predict(df)[0])
    st.markdown(f'<div class="pred-box">${pred:,.2f}</div>', unsafe_allow_html=True)

# ============================================================
# 9. FEATURE IMPORTANCE (Sensitivity)
# ============================================================

st.subheader("📌 Feature Sensitivity (What drives this prediction?)")

base = float(model.predict(df)[0])
importance = {}

for col in df.columns:
    mod_df = df.copy()
    mod_df[col] *= 1.10
    importance[col] = abs(float(model.predict(mod_df)[0]) - base)

importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

fig, ax = plt.subplots()
ax.barh(list(importance.keys()), list(importance.values()), color="#38BDF8")
ax.invert_yaxis()
st.pyplot(fig)

st.write(ai_insight(
    "Feature Sensitivity",
    "Which input variables affect sales the most?",
    importance
))

# ============================================================
# 10. 10-WEEK FORECAST
# ============================================================

st.subheader("📈 10-Week Forecast")

future_weeks = np.arange(week, week + 10).astype(float)

df_future = df.loc[df.index.repeat(10)].copy()
df_future["Week"] = future_weeks

preds = model.predict(df_future)

fig2, ax2 = plt.subplots()
ax2.plot(future_weeks, preds, marker="o", color="#7dd3fc")
ax2.set_title("10-Week Predicted Sales Trend")
st.pyplot(fig2)

st.write(ai_insight(
    "10-Week Forecast",
    "Expected future weekly sales trend.",
    {"weeks": list(future_weeks), "sales": list(preds)}
))

st.markdown("</div>", unsafe_allow_html=True)
