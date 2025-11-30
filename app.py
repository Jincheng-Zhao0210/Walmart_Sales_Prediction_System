# ============================================================
# Walmart Weekly Sales Prediction Dashboard
# Dark Theme • Sky Blue Accents • AI Insights • Python 3.12 Compatible
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
            color: #F8FAFC;
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
        h1, h2, h3, h4, h5 {{
            color: #38BDF8 !important;
            font-weight: 800 !important;
        }}
        p, li {{
            color: rgba(255,255,255,0.95);
            font-size: 17px;
            line-height: 1.5;
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
    You are a business analyst explaining results to Walmart executives.
    Use clear business English, avoid ML jargon, and give 3–5 bullet points.

    Title: {title}
    Context: {explanation}
    Values: {values}
    """

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"(AI insight unavailable: {e})"

# ============================================================
# 4) HEADER
# ============================================================

st.markdown(
    "<h1 style='text-align:center;'>📊 Walmart Weekly Sales Prediction Dashboard</h1>",
    unsafe_allow_html=True,
)

st.markdown(
    "<p style='text-align:center;color:#A5F3FC;'>Dark Theme • Sky Blue Accents • AI Insights</p>",
    unsafe_allow_html=True,
)

if logo64:
    st.markdown(
        f"<div style='text-align:center;'><img src='data:image/png;base64,{logo64}' width='150'></div>",
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# 5) BEAUTIFUL PROJECT OVERVIEW SECTION
# ============================================================

st.markdown("""
<div style="background: rgba(0,0,0,0.65); padding: 25px; 
            border-radius: 15px; margin-bottom: 25px;
            border: 1px solid rgba(56, 189, 248, 0.45);">

<h2 style="text-align:center;">📘 Project Overview</h2>

<p>
This dashboard is part of a machine learning project designed to predict 
<strong>Walmart's weekly sales</strong> and provide insights that help 
leaders make faster and smarter decisions.
</p>

<h4>👥 For Store & Regional Managers</h4>
<ul>
    <li>📅 Determine how many employees to schedule</li>
    <li>📦 Decide how much inventory to order</li>
    <li>📈 Prepare for high-demand or seasonal peaks</li>
</ul>

<h4>🚚 For Supply Chain & Inventory Planners</h4>
<ul>
    <li>❌ Prevent stockouts and lost revenue</li>
    <li>📉 Reduce overstock and holding costs</li>
    <li>🚛 Plan replenishment more efficiently</li>
</ul>

<p>
By converting raw data into <strong>actionable insights</strong>, this application 
enhances operational forecasting, planning, and decision-making across Walmart stores.
</p>

</div>
""", unsafe_allow_html=True)

# ============================================================
# 6) MAIN INPUT CARD
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
    week = st.number_input("Week of Year", 1, 53, 1)
    fuel = st.number_input("Fuel Price ($)", value=2.50)
    cpi = st.number_input("CPI", value=220.0)
    unemp = st.number_input("Unemployment Rate (%)", value=5.0)

# ============================================================
# 7) FIXED INPUT SCHEMA — MATCHES YOUR MODEL EXACTLY
# ============================================================

input_df = pd.DataFrame({
    "Store": [float(store)],
    "Holiday_Flag": [float(holiday)],
    "Temperature": [float(temp)],
    "Fuel_Price": [float(fuel)],
    "CPI": [float(cpi)],
    "Unemployment": [float(unemp)],
    "Year": [float(year)],
    "Month": [float(month)],
    "Week": [float(week)],
})

# ============================================================
# 8) PREDICTION
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
            "Explain the meaning of this forecast for labor planning and inventory decisions.",
            {"predicted_sales": pred, "store": store},
        )
    )

# ============================================================
# 9) FEATURE SENSITIVITY
# ============================================================

st.subheader("📍 Feature Sensitivity (What drives this prediction?)")

base_pred = float(model.predict(input_df)[0])
importance = {}

for col in input_df.columns:
    tmp = input_df.copy()
    tmp[col] = tmp[col] * 1.10
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
        "Explain which inputs have the greatest influence on projected sales.",
        importance,
    )
)

# ============================================================
# 10) 10-WEEK FORECAST
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
        "Explain how this forecast supports planning and operations.",
        {"weeks": list(future_weeks), "sales": list(map(float, future_preds))},
    )
)

st.markdown("</div>", unsafe_allow_html=True)
