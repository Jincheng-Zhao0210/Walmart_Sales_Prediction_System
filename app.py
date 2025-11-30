# ============================================================
# Walmart Weekly Sales Prediction – Databricks MLflow Version
# Loads ensemble model from Databricks MLflow Registry
# + OpenAI business insights
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

# IMPORTANT — This model signature DOES NOT include Dept
MODEL_URI = "models:/workspace.jzhao221.walmart/1"


@st.cache_resource
def load_model():
    return mlflow.pyfunc.load_model(MODEL_URI)


model = load_model()


# ============================================================
# 2) HELPER — LOAD BACKGROUND / LOGO
# ============================================================

def load_image_base64(path: str):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return None


bg64 = load_image_base64("background.jpg")
logo64 = load_image_base64("logo.png")


# ============================================================
# 3) OPENAI CLIENT + INSIGHT FUNCTION
# ============================================================

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


def ai_insight(title: str, explanation: str, values: dict):
    prompt = f"""
    You are a business analyst explaining results to a Walmart regional manager.
    Use simple business English. No ML jargon.

    Title: {title}
    Context: {explanation}
    Values: {values}

    Provide 3–5 bullet points of insight.
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
# 4) PAGE STYLING
# ============================================================

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
            backdrop-filter: blur(10px);
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
# 5) HEADER
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
# 6) MAIN CARD + INPUT FORM
# ============================================================

st.markdown("<div class='main-card'>", unsafe_allow_html=True)

st.header("🔧 Enter Store & Week Information")

col1, col2 = st.columns(2)

with col1:
    store = st.number_input("Store ID", 1, 50, 1)
    holiday = st.selectbox("Holiday Flag (0 = No, 1 = Yes)", [0, 1])
    year = st.number_input("Year", 2010, 2030, 2023)

with col2:
    month = st.number_input("Month", 1, 12, 1)
    week = st.number_input("Week of Year", 1, 53, 1)
    temp = st.number_input("Temperature (°F)", value=70.0)
    fuel = st.number_input("Fuel Price ($)", value=2.50)
    cpi = st.number_input("CPI", value=220.0)
    unemp = st.number_input("Unemployment Rate (%)", value=5.0)

# IMPORTANT: Dept removed — matches MLflow signature
input_df = pd.DataFrame(
    {
        "Store": [store],
        "Holiday_Flag": [holiday],
        "Temperature": [temp],
        "Fuel_Price": [fuel],
        "CPI": [cpi],
        "Unemployment": [unemp],
        "Year": [year],
        "Month": [month],
        "Week": [week],
    }
).astype(float)


# ============================================================
# 7) PREDICTION
# ============================================================

st.subheader("📌 Predicted Weekly Sales")

if st.button("Predict Weekly Sales"):
    pred = float(model.predict(input_df)[0])
    st.markdown(
        f"<div class='pred-box'><b>${pred:,.2f}</b></div>", unsafe_allow_html=True
    )

    st.write("### 🧠 AI Interpretation")
    st.info(
        ai_insight(
            "Weekly Sales Prediction",
            "Explain what this prediction means for staffing and inventory decisions.",
            {"predicted_sales": pred, "store": store},
        )
    )

# ============================================================
# 8) FEATURE SENSITIVITY
# ============================================================

st.subheader("📍 Feature Sensitivity (What drives this prediction?)")

try:
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
    ax_imp.set_xlabel("Change in Predicted Sales (absolute)")
    ax_imp.set_title("Feature Sensitivity (10% Increase)")
    st.pyplot(fig_imp)

    st.write("### 🧠 AI Insight on Drivers")
    st.info(
        ai_insight(
            "Feature Sensitivity",
            "Explain which features matter most for this prediction.",
            importance,
        )
    )

except Exception as e:
    st.error(f"Could not compute feature sensitivity: {e}")

# ============================================================
# 9) 10-WEEK FORECAST
# ============================================================

st.subheader("📈 10-Week Sales Forecast")

future_weeks = np.arange(week, week + 10)

forecast_df = input_df.loc[input_df.index.repeat(10)].copy()
forecast_df["Week"] = future_weeks
forecast_df = forecast_df.astype(float)

future_preds = model.predict(forecast_df)

fig_fore, ax_fore = plt.subplots()
ax_fore.plot(future_weeks, future_preds, marker="o", color="#00D5FF")
ax_fore.set_xlabel("Week Number")
ax_fore.set_ylabel("Predicted Weekly Sales")
ax_fore.set_title("10-Week Forecast for This Store")
st.pyplot(fig_fore)

st.write("### 🧠 AI Insight on Forecast")
st.info(
    ai_insight(
        "10-Week Forecast",
        "Explain how this forecast can help planning (labor, inventory, promos).",
        {"weeks": list(future_weeks), "sales": list(map(float, future_preds))},
    )
)

st.markdown("</div>", unsafe_allow_html=True)
