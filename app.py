# ============================================================
# Walmart Weekly Sales Prediction – Databricks MLflow Version
# Corrected schema · No Dept · Proper Forecast · AI Insights
# ============================================================

import os
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

import mlflow
import mlflow.pyfunc
from openai import OpenAI

# ============================================================
# 1) DATABRICKS + MLflow CONFIG
# ============================================================

DATABRICKS_HOST = st.secrets["DATABRICKS_HOST"]
DATABRICKS_TOKEN = st.secrets["DATABRICKS_TOKEN"]

os.environ["DATABRICKS_HOST"] = DATABRICKS_HOST
os.environ["DATABRICKS_TOKEN"] = DATABRICKS_TOKEN

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

MODEL_URI = "models:/workspace.jzhao221.walmart/1"

@st.cache_resource
def load_model():
    return mlflow.pyfunc.load_model(MODEL_URI)

model = load_model()

# ============================================================
# 2) BACKGROUND + LOGO
# ============================================================

def load_image_base64(path):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except:
        return None

bg64 = load_image_base64("background.jpg")
logo64 = load_image_base64("logo.png")

if bg64:
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{bg64}");
            background-size: cover;
            background-attachment: fixed;
        }}
        .main-card {{
            background: rgba(0,0,0,0.75);
            padding: 28px;
            border-radius: 18px;
            color: white;
        }}
        .pred-box {{
            background: #0EA5E9;
            padding: 18px;
            border-radius: 14px;
            font-size: 26px;
            text-align: center;
            color: white;
            margin-top: 15px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# 3) OPENAI CLIENT + INSIGHT FUNCTION
# ============================================================

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def ai_insight(title, explanation, values):
    prompt = f"""
    You are a business analyst explaining results to a Walmart regional manager.
    Use simple business English and avoid ML jargon.

    Title: {title}
    Context: {explanation}
    Values: {values}

    Provide 3–5 bullet points.
    """

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=220,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"(AI insight unavailable: {e})"


# ============================================================
# 4) HEADER
# ============================================================

st.markdown(
    "<h1 style='text-align:center;color:#38BDF8;'>Walmart Weekly Sales Predictor</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center;color:#93c5fd;'>Databricks MLflow Model · AI Insights · 10-Week Forecast</p>",
    unsafe_allow_html=True,
)

if logo64:
    st.markdown(
        f"<div style='text-align:center;'><img src='data:image/png;base64,{logo64}' width='160'></div>",
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# 5) MAIN INPUT CARD
# ============================================================

st.markdown("<div class='main-card'>", unsafe_allow_html=True)

st.header("🔧 Enter Store & Week Information")

col1, col2 = st.columns(2)

with col1:
    store = st.number_input("Store ID", 1, 50, 1)
    holiday = st.selectbox("Holiday Flag (0 = No, 1 = Yes)", [0, 1])
    year = st.number_input("Year", 2010, 2030, 2023)
    temp = st.number_input("Temperature (°F)", value=70.0)

with col2:
    month = st.number_input("Month", 1, 12, 1)
    week = st.number_input("Week", 1, 53, 1)
    fuel = st.number_input("Fuel Price ($)", value=2.50)
    cpi = st.number_input("CPI", value=220.0)
    unemp = st.number_input("Unemployment Rate (%)", value=5.0)

# ============================================================
# FIXED INPUT SCHEMA — matches your model EXACTLY
# ============================================================

input_df = pd.DataFrame(
    {
        "Store": [float(store)],
        "Holiday_Flag": [float(holiday)],
        "Temperature": [float(temp)],
        "Fuel_Price": [float(fuel)],
        "CPI": [float(cpi)],
        "Unemployment": [float(unemp)],
        "Year": [float(year)],
        "Month": [float(month)],
        "Week": [float(week)],
    }
)

# ============================================================
# 6) PREDICTION
# ============================================================

st.subheader("📌 Predicted Weekly Sales")

if st.button("Predict Weekly Sales"):
    pred = float(model.predict(input_df)[0])

    st.markdown(
        f"<div class='pred-box'><b>${pred:,.2f}</b></div>",
        unsafe_allow_html=True,
    )

    st.write("### 🧠 AI Interpretation")
    st.info(
        ai_insight(
            "Weekly Sales Prediction",
            "Explain what this means for labor, scheduling, and inventory decisions.",
            {"predicted_sales": pred, "store": store},
        )
    )

# ============================================================
# 7) FEATURE SENSITIVITY — fully fixed
# ============================================================

st.subheader("📍 Feature Sensitivity (What drives this prediction?)")

base_pred = float(model.predict(input_df)[0])
importance = {}

for col in input_df.columns:
    tmp = input_df.copy()
    tmp[col] = tmp[col] * 1.10  # +10%
    new_pred = float(model.predict(tmp)[0])
    importance[col] = abs(new_pred - base_pred)

importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

fig_imp, ax_imp = plt.subplots()
ax_imp.barh(list(importance.keys()), list(importance.values()), color="#38BDF8")
ax_imp.invert_yaxis()
ax_imp.set_xlabel("Change in Predicted Sales")
ax_imp.set_title("Feature Sensitivity (+10%)")

st.pyplot(fig_imp)

st.write("### 🧠 AI Insight on Drivers")
st.info(
    ai_insight(
        "Feature Sensitivity",
        "Explain which inputs most affect the prediction.",
        importance,
    )
)

# ============================================================
# 8) 10-WEEK FORECAST — fully fixed
# ============================================================

st.subheader("📈 10-Week Sales Forecast")

future_weeks = np.arange(float(week), float(week) + 10)

forecast_df = pd.DataFrame({
    "Store": [float(store)] * 10,
    "Holiday_Flag": [float(holiday)] * 10,
    "Temperature": [float(temp)] * 10,
    "Fuel_Price": [float(fuel)] * 10,
    "CPI": [float(cpi)] * 10,
    "Unemployment": [float(unemp)] * 10,
    "Year": [float(year)] * 10,
    "Month": [float(month)] * 10,
    "Week": future_weeks,
})

future_preds = model.predict(forecast_df)

fig_fore, ax_fore = plt.subplots()
ax_fore.plot(future_weeks, future_preds, marker="o", color="#00D5FF")
ax_fore.set_xlabel("Week Number")
ax_fore.set_ylabel("Predicted Weekly Sales")
ax_fore.set_title("10-Week Forecast")

st.pyplot(fig_fore)

st.write("### 🧠 AI Insight on Forecast")
st.info(
    ai_insight(
        "10-Week Forecast",
        "Explain how this helps managers plan ahead.",
        {"weeks": list(future_weeks), "sales": list(map(float, future_preds))},
    )
)

st.markdown("</div>", unsafe_allow_html=True)
