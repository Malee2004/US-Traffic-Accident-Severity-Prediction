"""
US Traffic Accident Severity Predictor — Streamlit Dashboard
==============================================================
Loads the trained model (best_severity_model.pkl, produced by Part_B_ML.ipynb)
and lets a user enter weather / time / road-infrastructure conditions to get
a predicted accident Severity (1-4) with class probabilities.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py
"""

import os
import io
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin

# ----------------------------------------------------------------------
# IMPORTANT: this custom transformer class must be redefined here,
# identically to how it was defined in the training notebook, so that
# joblib/pickle can resolve it when loading best_severity_model.pkl.
# ----------------------------------------------------------------------
class BooleanToIntTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.Series):
            return X.replace({True: 1, False: 0}).to_frame()
        return X.replace({True: 1, False: 0})

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            raise ValueError("input_features must be provided to get_feature_names_out")
        return input_features


# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Accident Severity Predictor",
    page_icon="🚧",
    layout="wide",
)

NUMERIC_FEATURES = [
    "Distance(mi)", "Temperature(F)", "Wind_Chill(F)", "Humidity(%)",
    "Pressure(in)", "Visibility(mi)", "Wind_Speed(mph)", "Precipitation(in)",
    "Hour", "Month", "Year",
]
CATEGORICAL_FEATURES = ["DayOfWeek"]
BOOLEAN_FEATURES = [
    "Amenity", "Bump", "Crossing", "Give_Way", "Junction", "No_Exit",
    "Railway", "Roundabout", "Station", "Stop", "Traffic_Calming",
    "Traffic_Signal", "Is_Weekend", "Is_Rush_Hour", "Is_Night",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BOOLEAN_FEATURES

DAYS_OF_WEEK = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

SEVERITY_INFO = {
    1: ("Minor", "#2ecc71"),
    2: ("Moderate", "#f1c40f"),
    3: ("Serious", "#e67e22"),
    4: ("Severe", "#e74c3c"),
}


# ----------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------
@st.cache_resource
def load_model(path_or_buffer):
    return joblib.load(path_or_buffer)


def decode_severity(raw_label):
    """
    Model pipelines in this project were trained two ways:
      - Logistic Regression / Random Forest -> fit on raw Severity (1-4)
      - XGBoost -> fit on label-encoded Severity (0-3)
    This normalises either output back to the real 1-4 severity scale.
    """
    raw_label = int(raw_label)
    if raw_label in (0, 1, 2, 3) and raw_label not in (1, 2, 3, 4):
        return raw_label + 1
    if raw_label == 0:
        return 1
    return raw_label


def get_class_probabilities(model, input_df):
    if not hasattr(model, "predict_proba"):
        return None
    proba = model.predict_proba(input_df)[0]
    raw_classes = model.classes_
    labels = [decode_severity(c) for c in raw_classes]
    return dict(zip(labels, proba))


# ----------------------------------------------------------------------
# Sidebar — model source + info
# ----------------------------------------------------------------------
st.sidebar.title("🚧 Model")

default_path = "best_severity_model.pkl"
model = None
model_source = None

if os.path.exists(default_path):
    model = load_model(default_path)
    model_source = default_path
else:
    uploaded_model = st.sidebar.file_uploader("Upload best_severity_model.pkl", type=["pkl"])
    if uploaded_model is not None:
        model = load_model(io.BytesIO(uploaded_model.getvalue()))
        model_source = uploaded_model.name

if model is not None:
    st.sidebar.success(f"Model loaded from: {model_source}")
    model_step = model.named_steps.get("model") if hasattr(model, "named_steps") else None
    if model_step is not None:
        st.sidebar.write("**Algorithm:**", type(model_step).__name__)
else:
    st.sidebar.warning("No model loaded yet — upload best_severity_model.pkl above.")

st.sidebar.markdown("---")
st.sidebar.caption(
    "Trained on US Accidents data (Part A cleaning + Part B modeling). "
    "Predicts Severity 1 (minor) to 4 (severe) from weather, time, and "
    "road-infrastructure conditions."
)

# Session history of predictions
if "history" not in st.session_state:
    st.session_state.history = []


# ----------------------------------------------------------------------
# Main layout
# ----------------------------------------------------------------------
st.title("US Traffic Accident — Severity Predictor")
st.write("Enter the conditions below and get a predicted accident severity.")

col_input, col_result = st.columns([1.3, 1])

with col_input:
    st.subheader("Conditions")

    with st.form("prediction_form"):
        t1, t2, t3 = st.tabs(["Weather", "Time", "Road Infrastructure"])

        with t1:
            c1, c2 = st.columns(2)
            with c1:
                temperature = st.number_input("Temperature (F)", -30.0, 130.0, 65.0)
                wind_chill = st.number_input("Wind Chill (F)", -50.0, 130.0, 60.0)
                humidity = st.slider("Humidity (%)", 0, 100, 60)
                pressure = st.number_input("Pressure (in)", 20.0, 35.0, 29.9)
            with c2:
                visibility = st.number_input("Visibility (mi)", 0.0, 60.0, 10.0)
                wind_speed = st.number_input("Wind Speed (mph)", 0.0, 120.0, 8.0)
                precipitation = st.number_input("Precipitation (in)", 0.0, 10.0, 0.0)
                distance = st.number_input("Distance affected (mi)", 0.0, 50.0, 0.5)

        with t2:
            c1, c2 = st.columns(2)
            with c1:
                hour = st.slider("Hour of day (0-23)", 0, 23, 8)
                day_of_week = st.selectbox("Day of Week", DAYS_OF_WEEK)
            with c2:
                month = st.selectbox("Month", list(range(1, 13)), index=5)
                year = st.number_input("Year", 2016, 2030, datetime.now().year, step=1)

            is_weekend = day_of_week in ("Saturday", "Sunday")
            is_rush_hour = hour in (7, 8, 9, 16, 17, 18)
            is_night = (hour >= 20) or (hour < 6)
            st.caption(
                f"Auto-derived → Is_Weekend: **{is_weekend}**, "
                f"Is_Rush_Hour: **{is_rush_hour}**, Is_Night: **{is_night}**"
            )

        with t3:
            c1, c2, c3 = st.columns(3)
            with c1:
                amenity = st.checkbox("Amenity")
                bump = st.checkbox("Bump")
                crossing = st.checkbox("Crossing")
                give_way = st.checkbox("Give Way")
                junction = st.checkbox("Junction")
            with c2:
                no_exit = st.checkbox("No Exit")
                railway = st.checkbox("Railway")
                roundabout = st.checkbox("Roundabout")
                station = st.checkbox("Station")
            with c3:
                stop = st.checkbox("Stop")
                traffic_calming = st.checkbox("Traffic Calming")
                traffic_signal = st.checkbox("Traffic Signal")

        submitted = st.form_submit_button("Predict Severity", use_container_width=True)

with col_result:
    st.subheader("Prediction")

    if submitted:
        if model is None:
            st.error("Load a model first (see sidebar).")
        else:
            input_row = {
                "Distance(mi)": distance,
                "Temperature(F)": temperature,
                "Wind_Chill(F)": wind_chill,
                "Humidity(%)": humidity,
                "Pressure(in)": pressure,
                "Visibility(mi)": visibility,
                "Wind_Speed(mph)": wind_speed,
                "Precipitation(in)": precipitation,
                "Hour": hour,
                "Month": month,
                "Year": year,
                "DayOfWeek": day_of_week,
                "Amenity": amenity,
                "Bump": bump,
                "Crossing": crossing,
                "Give_Way": give_way,
                "Junction": junction,
                "No_Exit": no_exit,
                "Railway": railway,
                "Roundabout": roundabout,
                "Station": station,
                "Stop": stop,
                "Traffic_Calming": traffic_calming,
                "Traffic_Signal": traffic_signal,
                "Is_Weekend": is_weekend,
                "Is_Rush_Hour": is_rush_hour,
                "Is_Night": is_night,
            }
            input_df = pd.DataFrame([input_row])[FEATURES]

            raw_pred = model.predict(input_df)[0]
            severity = decode_severity(raw_pred)
            label, color = SEVERITY_INFO.get(severity, ("Unknown", "#95a5a6"))

            st.markdown(
                f"""
                <div style="padding:1.2rem;border-radius:0.6rem;background-color:{color}22;
                            border:2px solid {color};text-align:center;">
                    <div style="font-size:0.9rem;color:#555;">Predicted Severity</div>
                    <div style="font-size:2.4rem;font-weight:700;color:{color};">
                        {severity} — {label}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            proba = get_class_probabilities(model, input_df)
            if proba:
                st.markdown("**Class probabilities**")
                proba_series = pd.Series(proba).sort_index()
                fig, ax = plt.subplots(figsize=(4, 2.5))
                colors = [SEVERITY_INFO.get(i, ("", "#95a5a6"))[1] for i in proba_series.index]
                ax.bar(proba_series.index.astype(str), proba_series.values, color=colors)
                ax.set_ylim(0, 1)
                ax.set_xlabel("Severity")
                ax.set_ylabel("Probability")
                st.pyplot(fig)

            st.session_state.history.append({**input_row, "Predicted_Severity": severity})
    else:
        st.info("Fill in the conditions and click **Predict Severity**.")


# ----------------------------------------------------------------------
# Prediction history
# ----------------------------------------------------------------------
st.markdown("---")
st.subheader("Session Prediction History")

if st.session_state.history:
    hist_df = pd.DataFrame(st.session_state.history)
    st.dataframe(hist_df, use_container_width=True)
    csv_bytes = hist_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download history as CSV",
        data=csv_bytes,
        file_name="severity_predictions_history.csv",
        mime="text/csv",
    )
    if st.button("Clear history"):
        st.session_state.history = []
        st.rerun()
else:
    st.caption("No predictions made yet this session.")
