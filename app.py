import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Accident Severity Intelligence",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>

.main {
    background-color: #0b1120;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.hero {
    padding: 25px;
    border-radius: 20px;
    background: linear-gradient(135deg, #111827, #172554);
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 42px;
    margin-bottom: 5px;
}

.hero p {
    color: #cbd5e1;
    font-size: 17px;
}

.metric-card {
    background: linear-gradient(145deg, #111827, #1e293b);
    padding: 20px;
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.08);
    text-align: center;
    min-height: 120px;
}

.metric-title {
    color: #94a3b8;
    font-size: 14px;
}

.metric-value {
    color: white;
    font-size: 28px;
    font-weight: 700;
    margin-top: 8px;
}

.section-title {
    font-size: 23px;
    font-weight: 700;
    margin-top: 20px;
    margin-bottom: 15px;
}

.result-box {
    padding: 25px;
    border-radius: 18px;
    background: linear-gradient(135deg, #172554, #1e3a8a);
    border: 1px solid rgba(255,255,255,0.12);
    text-align: center;
}

.result-title {
    color: #cbd5e1;
    font-size: 16px;
}

.result-value {
    color: white;
    font-size: 38px;
    font-weight: 800;
    margin-top: 8px;
}

.stButton > button {
    width: 100%;
    border-radius: 12px;
    height: 50px;
    font-size: 17px;
    font-weight: 700;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def load_model():
    return joblib.load("best_severity_model.pkl")


try:
    pipeline = load_model()
except Exception as e:
    st.error("❌ Could not load best_severity_model.pkl")
    st.info("Make sure best_severity_model.pkl is in the same folder as app.py")
    st.stop()


# =========================================================
# HEADER
# =========================================================
st.markdown("""
<div class="hero">

<h1>🚨 Accident Severity Intelligence</h1>

<p>
AI-powered US Traffic Accident Severity Prediction System
</p>

<p>
Predict accident severity using weather, road, traffic and
time-based conditions.
</p>

</div>
""", unsafe_allow_html=True)


# =========================================================
# MODEL INFORMATION
# =========================================================
with st.expander("🧠 Model Information", expanded=False):

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Model", "XGBoost")

    with c2:
        st.metric("Classes", "4")

    with c3:
        st.metric("Estimators", "200")

    with c4:
        st.metric("Learning Rate", "0.10")


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.markdown("## 🚗 Accident Information")
st.sidebar.caption("Enter the accident conditions below.")

# =========================================================
# ROAD INFORMATION
# =========================================================
st.sidebar.markdown("### 🛣️ Road & Location")

distance = st.sidebar.number_input(
    "Distance (miles)",
    min_value=0.0,
    max_value=1000.0,
    value=1.0,
    step=0.1
)

day_of_week = st.sidebar.selectbox(
    "Day of Week",
    [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ]
)

# =========================================================
# WEATHER
# =========================================================
st.sidebar.markdown("### 🌦️ Weather Conditions")

temperature = st.sidebar.number_input(
    "Temperature (°F)",
    min_value=-50.0,
    max_value=150.0,
    value=70.0,
    step=1.0
)

wind_chill = st.sidebar.number_input(
    "Wind Chill (°F)",
    min_value=-100.0,
    max_value=150.0,
    value=70.0,
    step=1.0
)

humidity = st.sidebar.slider(
    "Humidity (%)",
    min_value=0,
    max_value=100,
    value=60
)

pressure = st.sidebar.number_input(
    "Pressure (in)",
    min_value=20.0,
    max_value=35.0,
    value=29.9,
    step=0.1
)

visibility = st.sidebar.number_input(
    "Visibility (miles)",
    min_value=0.0,
    max_value=100.0,
    value=10.0,
    step=0.5
)

wind_speed = st.sidebar.number_input(
    "Wind Speed (mph)",
    min_value=0.0,
    max_value=200.0,
    value=8.0,
    step=1.0
)

precipitation = st.sidebar.number_input(
    "Precipitation (in)",
    min_value=0.0,
    max_value=20.0,
    value=0.0,
    step=0.01
)


# =========================================================
# TIME INFORMATION
# =========================================================
st.sidebar.markdown("### 🕐 Time Information")

hour = st.sidebar.slider(
    "Accident Hour",
    min_value=0,
    max_value=23,
    value=17
)

month = st.sidebar.slider(
    "Month",
    min_value=1,
    max_value=12,
    value=6
)

year = st.sidebar.number_input(
    "Year",
    min_value=2016,
    max_value=2035,
    value=2023,
    step=1
)

# Automatic time features
is_weekend = day_of_week in ["Saturday", "Sunday"]

is_rush_hour = (
    7 <= hour <= 9
    or
    16 <= hour <= 19
)

is_night = (
    hour >= 20
    or
    hour <= 5
)


# =========================================================
# ROAD / TRAFFIC CONDITIONS
# =========================================================
st.sidebar.markdown("### 🚦 Road & Traffic Conditions")

amenity = st.sidebar.checkbox("Amenity")
bump = st.sidebar.checkbox("Bump")
crossing = st.sidebar.checkbox("Crossing")
give_way = st.sidebar.checkbox("Give Way")
junction = st.sidebar.checkbox("Junction")
no_exit = st.sidebar.checkbox("No Exit")
railway = st.sidebar.checkbox("Railway")
roundabout = st.sidebar.checkbox("Roundabout")
station = st.sidebar.checkbox("Station")
stop = st.sidebar.checkbox("Stop")
traffic_calming = st.sidebar.checkbox("Traffic Calming")
traffic_signal = st.sidebar.checkbox("Traffic Signal")


# =========================================================
# CREATE INPUT DATAFRAME
# =========================================================
input_data = pd.DataFrame({

    "Distance(mi)": [distance],

    "Temperature(F)": [temperature],
    "Wind_Chill(F)": [wind_chill],
    "Humidity(%)": [humidity],
    "Pressure(in)": [pressure],
    "Visibility(mi)": [visibility],
    "Wind_Speed(mph)": [wind_speed],
    "Precipitation(in)": [precipitation],

    "Hour": [hour],
    "Month": [month],
    "Year": [year],

    "DayOfWeek": [day_of_week],

    "Amenity": [amenity],
    "Bump": [bump],
    "Crossing": [crossing],
    "Give_Way": [give_way],
    "Junction": [junction],
    "No_Exit": [no_exit],
    "Railway": [railway],
    "Roundabout": [roundabout],
    "Station": [station],
    "Stop": [stop],
    "Traffic_Calming": [traffic_calming],
    "Traffic_Signal": [traffic_signal],

    "Is_Weekend": [is_weekend],
    "Is_Rush_Hour": [is_rush_hour],
    "Is_Night": [is_night]
})


# =========================================================
# TOP METRICS
# =========================================================
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("""
    <div class="metric-card">
    <div class="metric-title">🌡️ Temperature</div>
    <div class="metric-value">""" + f"{temperature:.0f}°F" + """</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="metric-card">
    <div class="metric-title">💧 Humidity</div>
    <div class="metric-value">""" + f"{humidity}%" + """</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="metric-card">
    <div class="metric-title">👁️ Visibility</div>
    <div class="metric-value">""" + f"{visibility:.1f} mi" + """</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    period = "🌙 Night" if is_night else "☀️ Day"

    st.markdown("""
    <div class="metric-card">
    <div class="metric-title">🕐 Period</div>
    <div class="metric-value">""" + period + """</div>
    </div>
    """, unsafe_allow_html=True)


st.markdown("")


# =========================================================
# CURRENT CONDITIONS
# =========================================================
st.markdown('<div class="section-title">📡 Current Accident Conditions</div>',
            unsafe_allow_html=True)

condition1, condition2, condition3 = st.columns(3)

with condition1:

    if is_rush_hour:
        st.warning("🚦 **Rush Hour Detected**\n\nTraffic density may be higher.")
    else:
        st.success("✅ **Non-Rush Hour**\n\nNormal traffic period.")

with condition2:

    if is_night:
        st.warning("🌙 **Night-Time Accident**\n\nReduced visibility may increase risk.")
    else:
        st.success("☀️ **Day-Time Accident**\n\nDaylight conditions detected.")

with condition3:

    if precipitation > 0:
        st.warning("🌧️ **Precipitation Present**\n\nWet conditions detected.")
    else:
        st.success("☀️ **No Precipitation**\n\nDry weather condition.")


# =========================================================
# INPUT SUMMARY
# =========================================================
with st.expander("📋 View Complete Input Data"):

    st.dataframe(
        input_data.T.rename(columns={0: "Value"}),
        use_container_width=True
    )


# =========================================================
# PREDICTION BUTTON
# =========================================================
st.markdown("---")

predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])

with predict_col2:

    predict_button = st.button(
        "🚨 ANALYZE ACCIDENT SEVERITY",
        type="primary",
        use_container_width=True
    )


# =========================================================
# PREDICTION
# =========================================================
if predict_button:

    try:

        # Pipeline handles preprocessing automatically
        prediction = pipeline.predict(input_data)

        predicted_class = int(prediction[0])

        # -------------------------------------------------
        # Convert model class to display severity
        # -------------------------------------------------
        #
        # If model classes are 0,1,2,3:
        # display as Severity 1,2,3,4
        #
        if predicted_class in [0, 1, 2, 3]:
            display_severity = predicted_class + 1
        else:
            display_severity = predicted_class

        # -------------------------------------------------
        # Probability
        # -------------------------------------------------
        probabilities = None

        try:
            probabilities = pipeline.predict_proba(input_data)[0]
        except Exception:
            probabilities = None

        # -------------------------------------------------
        # Risk classification
        # -------------------------------------------------
        if display_severity == 1:

            risk_level = "LOW"
            risk_icon = "🟢"
            risk_message = (
                "The model predicts a relatively low-severity accident."
            )

        elif display_severity == 2:

            risk_level = "MODERATE"
            risk_icon = "🟡"
            risk_message = (
                "The model predicts a moderate-severity accident."
            )

        elif display_severity == 3:

            risk_level = "HIGH"
            risk_icon = "🟠"
            risk_message = (
                "The model predicts a high-severity accident."
            )

        else:

            risk_level = "CRITICAL"
            risk_icon = "🔴"
            risk_message = (
                "The model predicts a critical/highest-severity accident."
            )


        # =================================================
        # RESULT
        # =================================================
        st.markdown("---")

        st.markdown(
            f"""
            <div class="result-box">

            <div class="result-title">
            🚨 PREDICTED ACCIDENT SEVERITY
            </div>

            <div class="result-value">
            Severity {display_severity}
            </div>

            <div style="font-size:22px; margin-top:12px;">
            {risk_icon} {risk_level} RISK
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


        st.markdown("")


        # =================================================
        # RESULT METRICS
        # =================================================
        r1, r2, r3 = st.columns(3)

        with r1:

            st.metric(
                "Predicted Severity",
                f"Level {display_severity}"
            )

        with r2:

            st.metric(
                "Risk Level",
                risk_level
            )

        with r3:

            if probabilities is not None:
                confidence = float(np.max(probabilities)) * 100

                st.metric(
                    "Model Confidence",
                    f"{confidence:.2f}%"
                )

            else:

                st.metric(
                    "Model",
                    "XGBoost"
                )


        st.info(risk_message)


        # =================================================
        # PROBABILITY CHART
        # =================================================
        if probabilities is not None:

            st.markdown(
                '<div class="section-title">📊 Severity Probability Analysis</div>',
                unsafe_allow_html=True
            )

            severity_labels = [
                "Severity 1",
                "Severity 2",
                "Severity 3",
                "Severity 4"
            ]

            probability_percent = probabilities * 100

            fig = go.Figure()

            fig.add_trace(
                go.Bar(
                    x=severity_labels,
                    y=probability_percent,
                    text=[
                        f"{x:.2f}%"
                        for x in probability_percent
                    ],
                    textposition="outside"
                )
            )

            fig.update_layout(
                title="Prediction Probability Distribution",
                yaxis_title="Probability (%)",
                xaxis_title="Accident Severity",
                yaxis=dict(
                    range=[
                        0,
                        max(100, max(probability_percent) + 15)
                    ]
                ),
                template="plotly_dark",
                height=430,
                margin=dict(
                    l=20,
                    r=20,
                    t=60,
                    b=20
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


        # =================================================
        # RISK INDICATORS
        # =================================================
        st.markdown(
            '<div class="section-title">🔎 Risk Indicators</div>',
            unsafe_allow_html=True
        )

        risk1, risk2, risk3, risk4 = st.columns(4)


        # Visibility
        with risk1:

            if visibility < 3:

                st.error(
                    "👁️ **Low Visibility**\n\n"
                    "Visibility is below 3 miles."
                )

            elif visibility < 6:

                st.warning(
                    "👁️ **Reduced Visibility**\n\n"
                    "Visibility is moderate."
                )

            else:

                st.success(
                    "👁️ **Good Visibility**\n\n"
                    "Visibility is relatively good."
                )


        # Weather
        with risk2:

            if precipitation > 0.1:

                st.error(
                    "🌧️ **Heavy Precipitation**\n\n"
                    "Wet conditions detected."
                )

            elif precipitation > 0:

                st.warning(
                    "🌦️ **Precipitation**\n\n"
                    "Some precipitation detected."
                )

            else:

                st.success(
                    "☀️ **Dry Conditions**\n\n"
                    "No precipitation."
                )


        # Traffic
        with risk3:

            if is_rush_hour:

                st.warning(
                    "🚦 **Rush Hour**\n\n"
                    "Higher traffic activity."
                )

            else:

                st.success(
                    "🚦 **Normal Traffic Period**\n\n"
                    "Outside rush hour."
                )


        # Road features
        with risk4:

            infrastructure_count = sum([
                junction,
                railway,
                crossing,
                traffic_signal,
                stop,
                roundabout
            ])

            if infrastructure_count >= 3:

                st.warning(
                    "🛣️ **Complex Road Environment**\n\n"
                    f"{infrastructure_count} infrastructure "
                    "features selected."
                )

            else:

                st.success(
                    "🛣️ **Normal Road Environment**\n\n"
                    "Limited infrastructure complexity."
                )


        # =================================================
        # PREDICTION SUMMARY
        # =================================================
        st.markdown(
            '<div class="section-title">📝 Prediction Summary</div>',
            unsafe_allow_html=True
        )

        summary_data = pd.DataFrame({
            "Parameter": [
                "Predicted Severity",
                "Risk Level",
                "Day",
                "Hour",
                "Weekend",
                "Rush Hour",
                "Night",
                "Temperature",
                "Humidity",
                "Visibility",
                "Precipitation",
                "Distance"
            ],

            "Value": [
                f"Severity {display_severity}",
                risk_level,
                day_of_week,
                f"{hour}:00",
                "Yes" if is_weekend else "No",
                "Yes" if is_rush_hour else "No",
                "Yes" if is_night else "No",
                f"{temperature:.1f} °F",
                f"{humidity}%",
                f"{visibility:.1f} miles",
                f"{precipitation:.2f} in",
                f"{distance:.2f} miles"
            ]
        })

        st.dataframe(
            summary_data,
            use_container_width=True,
            hide_index=True
        )


        # =================================================
        # MODEL PROBABILITY TABLE
        # =================================================
        if probabilities is not None:

            st.markdown(
                '<div class="section-title">🎯 Model Confidence Breakdown</div>',
                unsafe_allow_html=True
            )

            probability_table = pd.DataFrame({
                "Severity": severity_labels,
                "Probability": [
                    f"{p:.2f}%"
                    for p in probability_percent
                ]
            })

            st.dataframe(
                probability_table,
                use_container_width=True,
                hide_index=True
            )


        # =================================================
        # FINAL ALERT
        # =================================================
        if display_severity >= 4:

            st.error(
                "🚨 **CRITICAL ALERT**\n\n"
                "The model has classified this scenario as "
                "the highest severity category."
            )

        elif display_severity == 3:

            st.warning(
                "⚠️ **HIGH-RISK SCENARIO**\n\n"
                "The predicted severity is high. "
                "Extra caution is recommended."
            )

        elif display_severity == 2:

            st.info(
                "ℹ️ **MODERATE-RISK SCENARIO**\n\n"
                "The predicted accident severity is moderate."
            )

        else:

            st.success(
                "✅ **LOW-RISK SCENARIO**\n\n"
                "The model predicts the lowest severity category."
            )


    except Exception as e:

        st.error("❌ Prediction failed.")

        st.code(str(e))

        st.warning(
            "Please check that the feature names and data types "
            "match those used during model training."
        )


# =========================================================
# FOOTER
# =========================================================
st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; color:#64748b; padding:15px;">

    <b>🚨 Accident Severity Intelligence Dashboard</b><br>

    Machine Learning • XGBoost • Traffic Accident Analytics

    </div>
    """,
    unsafe_allow_html=True
)
