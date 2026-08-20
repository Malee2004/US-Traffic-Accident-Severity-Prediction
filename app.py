import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

from sklearn.base import BaseEstimator, TransformerMixin


# =========================================================
# CUSTOM TRANSFORMER
# =========================================================
class BooleanToIntTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        if isinstance(X, pd.Series):
            return X.map({True: 1, False: 0}).to_frame()

        if isinstance(X, pd.DataFrame):
            return X.map(
                lambda x: 1 if x is True else 0 if x is False else x
            )

        return X

    def get_feature_names_out(self, input_features=None):

        if input_features is None:
            return np.array([], dtype=object)

        return np.asarray(input_features, dtype=object)


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Accident Severity Intelligence",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CSS
# =========================================================
st.markdown(
    """
    <style>

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    .hero {
        padding: 28px;
        border-radius: 20px;
        margin-bottom: 22px;
        background: linear-gradient(135deg, #111827, #172554);
        border: 1px solid rgba(255,255,255,.10);
    }

    .hero h1 {
        font-size: 40px;
        margin: 0 0 8px 0;
    }

    .hero p {
        color: #cbd5e1;
        font-size: 16px;
        margin: 4px 0;
    }

    .card {
        padding: 18px;
        border-radius: 16px;
        background: linear-gradient(145deg, #111827, #1e293b);
        border: 1px solid rgba(255,255,255,.08);
        text-align: center;
        min-height: 105px;
    }

    .card-title {
        color: #94a3b8;
        font-size: 13px;
    }

    .card-value {
        color: white;
        font-size: 27px;
        font-weight: 700;
        margin-top: 7px;
    }

    .section {
        font-size: 22px;
        font-weight: 700;
        margin: 22px 0 12px;
    }

    .result {
        padding: 28px;
        border-radius: 20px;
        text-align: center;
        background: linear-gradient(135deg, #172554, #1e3a8a);
        border: 1px solid rgba(255,255,255,.12);
    }

    .result-small {
        color: #cbd5e1;
        font-size: 15px;
    }

    .result-big {
        color: white;
        font-size: 40px;
        font-weight: 800;
        margin: 7px 0;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def load_pipeline():
    return joblib.load("best_severity_model.pkl")


try:

    pipeline = load_pipeline()

except FileNotFoundError:

    st.error("❌ best_severity_model.pkl was not found.")

    st.info(
        "Make sure best_severity_model.pkl is in the same "
        "GitHub repository folder as app.py."
    )

    st.stop()

except Exception as e:

    st.error("❌ Could not load best_severity_model.pkl")
    st.code(f"{type(e).__name__}: {str(e)}")
    st.stop()


# =========================================================
# HEADER
# =========================================================
st.markdown(
    """
    <div class="hero">

    <h1>🚨 Accident Severity Intelligence</h1>

    <p>
    AI-powered US Traffic Accident Severity Prediction System
    </p>

    <p>
    Analyze weather, road, traffic and time conditions
    to predict accident severity.
    </p>

    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# MODEL INFORMATION
# =========================================================
with st.expander("🧠 Model Information"):

    a, b, c, d = st.columns(4)

    a.metric("Algorithm", "XGBoost")
    b.metric("Classes", "4")
    c.metric("Estimators", "200")
    d.metric("Learning Rate", "0.10")


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("🚗 Accident Inputs")

st.sidebar.caption(
    "Enter the accident conditions below."
)


# =========================================================
# ROAD & TIME
# =========================================================
st.sidebar.markdown("### 🛣️ Road & Time")

distance = st.sidebar.number_input(
    "Distance (mi)",
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

hour = st.sidebar.slider(
    "Hour",
    0,
    23,
    17
)

month = st.sidebar.slider(
    "Month",
    1,
    12,
    6
)

year = st.sidebar.number_input(
    "Year",
    2016,
    2035,
    2023,
    1
)


# =========================================================
# AUTOMATIC TIME FEATURES
# =========================================================
is_weekend = day_of_week in [
    "Saturday",
    "Sunday"
]

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
# WEATHER
# =========================================================
st.sidebar.markdown("### 🌦️ Weather")

temperature = st.sidebar.number_input(
    "Temperature (°F)",
    -50.0,
    150.0,
    70.0,
    1.0
)

wind_chill = st.sidebar.number_input(
    "Wind Chill (°F)",
    -100.0,
    150.0,
    70.0,
    1.0
)

humidity = st.sidebar.slider(
    "Humidity (%)",
    0,
    100,
    60
)

pressure = st.sidebar.number_input(
    "Pressure (in)",
    20.0,
    35.0,
    29.9,
    0.1
)

visibility = st.sidebar.number_input(
    "Visibility (mi)",
    0.0,
    100.0,
    10.0,
    0.5
)

wind_speed = st.sidebar.number_input(
    "Wind Speed (mph)",
    0.0,
    200.0,
    8.0,
    1.0
)

precipitation = st.sidebar.number_input(
    "Precipitation (in)",
    0.0,
    20.0,
    0.0,
    0.01
)


# =========================================================
# ROAD / TRAFFIC
# =========================================================
st.sidebar.markdown("### 🚦 Road & Traffic")

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
# INPUT DATA
# =========================================================
input_df = pd.DataFrame({

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
# CURRENT CONDITIONS
# =========================================================
st.markdown(
    '<div class="section">📡 Current Conditions</div>',
    unsafe_allow_html=True
)

c1, c2, c3, c4 = st.columns(4)


with c1:

    st.markdown(
        f"""
        <div class="card">

        <div class="card-title">
        🌡️ Temperature
        </div>

        <div class="card-value">
        {temperature:.0f}°F
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with c2:

    st.markdown(
        f"""
        <div class="card">

        <div class="card-title">
        💧 Humidity
        </div>

        <div class="card-value">
        {humidity}%
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with c3:

    st.markdown(
        f"""
        <div class="card">

        <div class="card-title">
        👁️ Visibility
        </div>

        <div class="card-value">
        {visibility:.1f} mi
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with c4:

    period = "🌙 Night" if is_night else "☀️ Day"

    st.markdown(
        f"""
        <div class="card">

        <div class="card-title">
        🕐 Period
        </div>

        <div class="card-value">
        {period}
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# CONDITION ALERTS
# =========================================================
x1, x2, x3 = st.columns(3)


with x1:

    if is_rush_hour:
        st.warning("🚦 **Rush Hour Detected**")
    else:
        st.success("✅ **Non-Rush Hour**")


with x2:

    if precipitation > 0:
        st.warning("🌧️ **Precipitation Present**")
    else:
        st.success("☀️ **Dry Conditions**")


with x3:

    if is_weekend:
        st.info("📅 **Weekend**")
    else:
        st.info("📅 **Weekday**")


# =========================================================
# INPUT PREVIEW
# =========================================================
with st.expander("📋 View Input Data"):

    st.dataframe(
        input_df.T.rename(
            columns={0: "Value"}
        ),
        use_container_width=True
    )


# =========================================================
# PREDICTION
# =========================================================
st.markdown(
    '<div class="section">🎯 Prediction</div>',
    unsafe_allow_html=True
)


if st.button(
    "🚨 ANALYZE ACCIDENT SEVERITY",
    type="primary",
    use_container_width=True
):

    try:

        # =================================================
        # IMPORTANT:
        # Send the original DataFrame directly.
        # The saved pipeline should perform preprocessing.
        # =================================================
        prediction = pipeline.predict(input_df)

        raw_prediction = int(
            np.asarray(prediction).ravel()[0]
        )


        # =================================================
        # GET FINAL MODEL
        # =================================================
        model_step = None

        if hasattr(pipeline, "named_steps"):

            try:

                model_step = list(
                    pipeline.named_steps.values()
                )[-1]

            except Exception:

                model_step = None

        else:

            model_step = pipeline


        # =================================================
        # MODEL CLASSES
        # =================================================
        model_classes = getattr(
            model_step,
            "classes_",
            None
        )


        if model_classes is not None:

            model_classes = np.asarray(
                model_classes
            )

            matching = np.where(
                model_classes == raw_prediction
            )[0]

            if len(matching) > 0:

                class_index = int(
                    matching[0]
                )

            else:

                class_index = raw_prediction

        else:

            class_index = raw_prediction


        # =================================================
        # SEVERITY
        # =================================================
        if class_index in [0, 1, 2, 3]:

            severity = class_index + 1

        elif class_index in [1, 2, 3, 4]:

            severity = class_index

        else:

            severity = raw_prediction


        severity = int(
            max(1, min(4, severity))
        )


        # =================================================
        # PROBABILITY
        # =================================================
        probabilities = None

        try:

            probabilities = np.asarray(
                pipeline.predict_proba(
                    input_df
                )[0],
                dtype=float
            )

        except Exception:

            probabilities = None


        # =================================================
        # CONFIDENCE
        # =================================================
        if (
            probabilities is not None
            and len(probabilities) > 0
        ):

            confidence = (
                float(
                    np.max(probabilities)
                )
                * 100
            )

        else:

            confidence = None


        # =================================================
        # RISK MAP
        # =================================================
        risk_map = {

            1: (
                "🟢",
                "LOW",
                "The model predicts the lowest severity category."
            ),

            2: (
                "🟡",
                "MODERATE",
                "The model predicts a moderate severity accident."
            ),

            3: (
                "🟠",
                "HIGH",
                "The model predicts a high severity accident."
            ),

            4: (
                "🔴",
                "CRITICAL",
                "The model predicts the highest severity category."
            )

        }


        icon, risk, message = risk_map.get(
            severity,
            (
                "⚪",
                "UNKNOWN",
                "Prediction completed."
            )
        )


        # =================================================
        # RESULT
        # =================================================
        st.markdown(
            f"""
            <div class="result">

            <div class="result-small">
            🚨 PREDICTED ACCIDENT SEVERITY
            </div>

            <div class="result-big">
            Severity {severity}
            </div>

            <div style="font-size:22px;">
            {icon} {risk} RISK
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


        st.write("")


        r1, r2, r3 = st.columns(3)


        with r1:

            st.metric(
                "Predicted Severity",
                f"Level {severity}"
            )


        with r2:

            st.metric(
                "Risk Level",
                risk
            )


        with r3:

            if confidence is not None:

                st.metric(
                    "Model Confidence",
                    f"{confidence:.2f}%"
                )

            else:

                st.metric(
                    "Model Confidence",
                    "N/A"
                )


        st.info(message)


        # =================================================
        # PROBABILITY CHART
        # =================================================
        if (
            probabilities is not None
            and len(probabilities) == 4
        ):

            st.markdown(
                '<div class="section">📊 Severity Probability Distribution</div>',
                unsafe_allow_html=True
            )


            labels = [
                "Severity 1",
                "Severity 2",
                "Severity 3",
                "Severity 4"
            ]


            percentages = (
                probabilities * 100
            )


            fig = go.Figure()


            fig.add_trace(
                go.Bar(
                    x=labels,
                    y=percentages,
                    text=[
                        f"{p:.2f}%"
                        for p in percentages
                    ],
                    textposition="outside"
                )
            )


            fig.update_layout(
                template="plotly_dark",
                height=430,
                yaxis_title="Probability (%)",
                xaxis_title="Accident Severity",
                yaxis=dict(
                    range=[
                        0,
                        max(
                            100,
                            float(
                                max(percentages)
                            ) + 15
                        )
                    ]
                ),
                margin=dict(
                    l=20,
                    r=20,
                    t=40,
                    b=20
                )
            )


            st.plotly_chart(
                fig,
                use_container_width=True
            )


            probability_table = pd.DataFrame({

                "Severity": labels,

                "Probability": [
                    f"{p:.2f}%"
                    for p in percentages
                ]

            })


            st.dataframe(
                probability_table,
                use_container_width=True,
                hide_index=True
            )


        # =================================================
        # RISK INDICATORS
        # =================================================
        st.markdown(
            '<div class="section">🔎 Risk Indicators</div>',
            unsafe_allow_html=True
        )


        q1, q2, q3, q4 = st.columns(4)


        with q1:

            if visibility < 3:

                st.error(
                    "👁️ **Low Visibility**"
                )

            elif visibility < 6:

                st.warning(
                    "👁️ **Reduced Visibility**"
                )

            else:

                st.success(
                    "👁️ **Good Visibility**"
                )


        with q2:

            if precipitation > 0.1:

                st.error(
                    "🌧️ **Heavy Precipitation**"
                )

            elif precipitation > 0:

                st.warning(
                    "🌦️ **Precipitation**"
                )

            else:

                st.success(
                    "☀️ **Dry Conditions**"
                )


        with q3:

            if is_rush_hour:

                st.warning(
                    "🚦 **Rush Hour**"
                )

            else:

                st.success(
                    "🚦 **Normal Traffic Period**"
                )


        with q4:

            road_complexity = sum([
                junction,
                railway,
                crossing,
                traffic_signal,
                stop,
                roundabout
            ])


            if road_complexity >= 3:

                st.warning(
                    "🛣️ **Complex Road Environment**"
                )

            else:

                st.success(
                    "🛣️ **Normal Road Environment**"
                )


        # =================================================
        # SUMMARY
        # =================================================
        st.markdown(
            '<div class="section">📝 Prediction Summary</div>',
            unsafe_allow_html=True
        )


        summary = pd.DataFrame({

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

                f"Severity {severity}",
                risk,
                day_of_week,
                f"{hour}:00",
                "Yes" if is_weekend else "No",
                "Yes" if is_rush_hour else "No",
                "Yes" if is_night else "No",
                f"{temperature:.1f} °F",
                f"{humidity}%",
                f"{visibility:.1f} mi",
                f"{precipitation:.2f} in",
                f"{distance:.2f} mi"

            ]

        })


        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True
        )


        # =================================================
        # FINAL ALERT
        # =================================================
        if severity == 4:

            st.error(
                "🚨 **CRITICAL ALERT:** "
                "Highest severity category predicted."
            )

        elif severity == 3:

            st.warning(
                "⚠️ **HIGH-RISK SCENARIO:** "
                "High severity predicted."
            )

        elif severity == 2:

            st.info(
                "ℹ️ **MODERATE-RISK SCENARIO:** "
                "Moderate severity predicted."
            )

        else:

            st.success(
                "✅ **LOW-RISK SCENARIO:** "
                "Lowest severity predicted."
            )


    except Exception as e:

        # =================================================
        # DETAILED ERROR
        # =================================================
        st.error(
            "❌ Prediction failed."
        )

        st.code(
            f"{type(e).__name__}: {str(e)}"
        )

        st.warning(
            "The model file loaded successfully, but "
            "the saved preprocessing pipeline failed "
            "while processing the prediction input."
        )

        # Show useful debugging information
        with st.expander("🔧 Technical Debug Information"):

            st.write(
                "Input shape:",
                input_df.shape
            )

            st.write(
                "Input columns:",
                list(input_df.columns)
            )

            if hasattr(pipeline, "named_steps"):

                st.write(
                    "Pipeline steps:",
                    list(
                        pipeline.named_steps.keys()
                    )
                )

                for name, step in pipeline.named_steps.items():

                    st.write(
                        f"Step: {name}"
                    )

                    st.write(
                        "Type:",
                        str(type(step))
                    )

                    if hasattr(
                        step,
                        "feature_names_in_"
                    ):

                        st.write(
                            "Expected features:",
                            list(
                                step.feature_names_in_
                            )
                        )


# =========================================================
# FOOTER
# =========================================================
st.markdown("---")

st.markdown(
    """
    <div style="
        text-align:center;
        color:#64748b;
        padding:12px;
    ">

    <b>🚨 Accident Severity Intelligence Dashboard</b>
    <br>

    Machine Learning • XGBoost • Traffic Accident Analytics

    </div>
    """,
    unsafe_allow_html=True
)
