
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor


# --------------------------------------------------
# 1. PAGE SETTINGS
# --------------------------------------------------ō

st.set_page_config(
    page_title="Crime Hotspot Prediction",
    page_icon="📍",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

PRESETS = {
    "Chicago Loop": (41.8819, -87.6278),
    "Navy Pier": (41.8917, -87.6086),
    "Wrigleyville": (41.9484, -87.6553),
    "Hyde Park": (41.7943, -87.5907),
    "O'Hare Airport": (41.9742, -87.9073)
}

# The notebook's week-index formula assigns month 12
# to forecast week index 52 (starting 31 Dec 2022).
FORECAST_MONTH = 12


# --------------------------------------------------
# 2. LOAD MODEL AND DATA
# --------------------------------------------------

@st.cache_resource(show_spinner="Training prediction models...")
def load_models():

    with open(BASE_DIR / "meta.json", "r") as f:
        meta = json.load(f)

    features = meta["features"]
    threshold = float(meta["threshold"])

    train = pd.read_csv(BASE_DIR / "panel_train.csv")

    required = features + ["crime_count"]

    missing = [col for col in required if col not in train.columns]

    if missing:
        raise ValueError(
            f"Missing columns in panel_train.csv: {missing}"
        )

    train = train.dropna(subset=required)

    X_train = train[features]
    y_train = train["crime_count"]

    reg = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42
    )

    reg.fit(X_train, y_train)

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    )

    y_train_class = (y_train > threshold).astype(int)

    clf.fit(X_train, y_train_class)

    return reg, clf, features, threshold


@st.cache_data
def load_next_week():

    data = pd.read_csv(
        BASE_DIR / "next_week_input.csv",
        dtype={"cell_id": str}
    )

    required = [
        "cell_id", "cell_lat", "cell_lon",
        "lag_1", "lag_2", "lag_3", "lag_4",
        "roll_4_mean"
    ]

    missing = [col for col in required if col not in data.columns]

    if missing:
        raise ValueError(
            f"Missing columns in next_week_input.csv: {missing}"
        )

    for col in required[1:]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna(subset=required[1:]).reset_index(drop=True)

    if data.empty:
        raise ValueError(
            "No valid forecast rows found. Check next_week_input.csv."
        )

    return data


# --------------------------------------------------
# 3. PREDICTION FUNCTIONS
# --------------------------------------------------

def risk_label(probability):

    if probability >= 0.65:
        return "High"

    elif probability >= 0.35:
        return "Medium"

    return "Low"


def predict(df_in, reg, clf, features):

    X = df_in.copy()

    X["month"] = FORECAST_MONTH

    X = X[features]

    output = df_in.copy()

    output["predicted_crimes"] = np.clip(
        reg.predict(X), 0, None
    ).round(1)

    probabilities = clf.predict_proba(X)

    positive_class = list(clf.classes_).index(1)

    output["hotspot_probability"] = probabilities[
        :, positive_class
    ].round(3)

    output["risk_level"] = output[
        "hotspot_probability"
    ].apply(risk_label)

    return output


# --------------------------------------------------
# 4. LOAD APPLICATION DATA
# --------------------------------------------------

try:

    reg, clf, features, threshold = load_models()

    nxt = load_next_week()

    if 1 not in clf.classes_:
        st.error("Classifier has not learned the hotspot class.")
        st.stop()

except Exception as e:

    st.error(f"Application loading error: {e}")

    st.info(
        "Check whether app.py, meta.json, panel_train.csv "
        "and next_week_input.csv are in the same folder."
    )

    st.stop()


# --------------------------------------------------
# 5. HEADER AND SIDEBAR
# --------------------------------------------------

st.title("📍 Crime Hotspot Prediction")

st.caption(
    "B.Tech Data Science Minor Project | "
    "Swami Vivekananda University | "
    "Chicago Crime Data (2022) | 500 m Grid Cells"
)

with st.sidebar:

    st.header("⚙️ Settings")

    st.info(
        "Forecast period: Model week index 52 "
        "(starting 31 December 2022)."
    )

    st.markdown(
        f"**Forecast month feature:** December (12)"
    )

    st.markdown(
        f"**Hotspot rule:** More than {threshold:g} "
        "crimes per week in a 500 m area."
    )

    st.markdown("---")

    st.caption(
        "Risk levels: High ≥ 65%, "
        "Medium ≥ 35%, Low < 35%."
    )


# --------------------------------------------------
# 6. THREE MAIN TABS
# --------------------------------------------------

tab1, tab2, tab3 = st.tabs([
    "🗺️ City Hotspot Map",
    "📌 Check a Location",
    "🧮 What-if Calculator"
])


# ==================================================
# TAB 1: CITY HOTSPOT MAP
# ==================================================

with tab1:

    st.subheader("Predicted Crime Hotspots for Next Week")

    result = predict(nxt, reg, clf, features)

    k = st.slider(
        "Show the top K highest-risk areas",
        min_value=10,
        max_value=100,
        value=30,
        step=5
    )

    top = result.sort_values(
        "hotspot_probability",
        ascending=False
    ).head(k)

    # KPI CARDS

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Areas Analysed",
        f"{len(result):,}"
    )

    c2.metric(
        "Predicted Hotspots",
        f"{int((result['hotspot_probability'] >= 0.5).sum()):,}"
    )

    c3.metric(
        "Highest Predicted Crimes",
        f"{result['predicted_crimes'].max():.1f}"
    )

    st.markdown("---")

    # INTERACTIVE CITY MAP

    st.markdown("### Interactive Crime Hotspot Map")

    fig = px.scatter_map(
        result,
        lat="cell_lat",
        lon="cell_lon",
        color="hotspot_probability",
        color_continuous_scale="YlOrRd",
        range_color=(0, 1),
        zoom=9.3,
        height=600,
        map_style="open-street-map",
        hover_data={
            "cell_id": True,
            "predicted_crimes": True,
            "hotspot_probability": True,
            "risk_level": True,
            "cell_lat": False,
            "cell_lon": False
        },
        labels={
            "hotspot_probability": "Hotspot Probability",
            "predicted_crimes": "Predicted Crimes"
        }
    )

    fig.update_traces(
        marker=dict(size=9)
    )

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        coloraxis_colorbar=dict(
            title="Hotspot Probability"
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.caption(
        "Red and orange areas indicate higher predicted "
        "hotspot probability. Each point represents a 500 m grid cell."
    )

    # TOP K TABLE

    st.markdown(f"### Top {k} Highest-Risk Areas")

    show = top[
        [
            "cell_id",
            "cell_lat",
            "cell_lon",
            "predicted_crimes",
            "hotspot_probability",
            "risk_level"
        ]
    ]

    st.dataframe(
        show.reset_index(drop=True),
        use_container_width=True
    )

    # DOWNLOAD FORECAST

    st.download_button(
        label="⬇️ Download Full Forecast (CSV)",
        data=result.to_csv(index=False),
        file_name="hotspot_forecast.csv",
        mime="text/csv"
    )


# ==================================================
# TAB 2: CHECK A LOCATION
# ==================================================

with tab2:

    st.subheader("Check the Crime Risk of a Location")

    st.write(
        "Select a preset location or enter your own "
        "latitude and longitude."
    )

    # QUICK PICK STATE

    def update_quick_pick():

        selected = st.session_state.quick_pick

        if selected in PRESETS:

            coordinates = PRESETS[selected]

            st.session_state.location_lat = coordinates[0]
            st.session_state.location_lon = coordinates[1]


    if "location_lat" not in st.session_state:

        st.session_state.location_lat = 41.8819
        st.session_state.location_lon = -87.6278

    choice = st.selectbox(
        "Quick Pick",
        ["Type my own"] + list(PRESETS.keys()),
        key="quick_pick",
        on_change=update_quick_pick
    )

    a, b = st.columns(2)

    lat = a.number_input(
        "Latitude",
        min_value=41.60,
        max_value=42.10,
        format="%.4f",
        key="location_lat"
    )

    lon = b.number_input(
        "Longitude",
        min_value=-88.00,
        max_value=-87.50,
        format="%.4f",
        key="location_lon"
    )

    if st.button(
        "🔍 Predict for This Location",
        key="location_predict"
    ):

        # Calculate distance from every analysed cell

        distances = np.hypot(
            (nxt["cell_lat"].to_numpy() - lat) * 111.0,
            (nxt["cell_lon"].to_numpy() - lon)
            * 111.0
            * np.cos(np.radians(lat))
        )

        if len(distances) == 0 or not np.isfinite(distances).any():

            st.error("No valid analysed locations are available.")

        else:

            nearest_position = int(np.nanargmin(distances))

            dist = float(distances[nearest_position])

            if dist > 1.5:

                st.warning(
                    f"The nearest analysed area is {dist:.1f} km away. "
                    "No nearby analysed grid cell is available "
                    "for a reliable forecast."
                )

            else:

                selected_row = nxt.iloc[[nearest_position]]

                row = predict(
                    selected_row,
                    reg,
                    clf,
                    features
                ).iloc[0]

                st.success(
                    f"Nearest analysed area is "
                    f"{dist * 1000:.0f} metres away."
                )

                # LOCATION KPIs

                m1, m2, m3 = st.columns(3)

                m1.metric(
                    "Predicted Crimes",
                    f"{row['predicted_crimes']:.1f}"
                )

                m2.metric(
                    "Hotspot Probability",
                    f"{row['hotspot_probability'] * 100:.0f}%"
                )

                m3.metric(
                    "Risk Level",
                    row["risk_level"]
                )

                # LAST FOUR WEEKS

                st.markdown(
                    "### Crimes in This Area During the Last 4 Weeks"
                )

                history = pd.DataFrame({
                    "Week": [
                        "4 Weeks Ago",
                        "3 Weeks Ago",
                        "2 Weeks Ago",
                        "Last Week"
                    ],
                    "Crimes": [
                        row["lag_4"],
                        row["lag_3"],
                        row["lag_2"],
                        row["lag_1"]
                    ]
                })

                st.bar_chart(
                    history.set_index("Week")
                )

                # LOCATION MAP

                st.markdown("### Selected Location")

                location_map = pd.DataFrame({
                    "lat": [lat, row["cell_lat"]],
                    "lon": [lon, row["cell_lon"]]
                })

                st.map(location_map)


# ==================================================
# TAB 3: WHAT-IF CALCULATOR
# ==================================================

with tab3:

    st.subheader("What-if Crime Prediction Calculator")

    st.write(
        "Enter the number of crimes recorded in an area "
        "during the previous four weeks."
    )

    # FOUR WEEK INPUTS

    cols = st.columns(4)

    l4 = cols[0].number_input(
        "4 Weeks Ago",
        0, 200, 2,
        key="what_l4"
    )

    l3 = cols[1].number_input(
        "3 Weeks Ago",
        0, 200, 3,
        key="what_l3"
    )

    l2 = cols[2].number_input(
        "2 Weeks Ago",
        0, 200, 4,
        key="what_l2"
    )

    l1 = cols[3].number_input(
        "Last Week",
        0, 200, 5,
        key="what_l1"
    )

    # AREA COORDINATES

    p1, p2 = st.columns(2)

    wlat = p1.number_input(
        "Area Latitude",
        41.60, 42.10,
        41.8819,
        format="%.4f",
        key="wlat"
    )

    wlon = p2.number_input(
        "Area Longitude",
        -88.00, -87.50,
        -87.6278,
        format="%.4f",
        key="wlon"
    )

    if st.button(
        "🧮 Calculate Forecast",
        key="whatif_predict"
    ):

        # CREATE INPUT ROW

        row = pd.DataFrame([{
            "lag_1": l1,
            "lag_2": l2,
            "lag_3": l3,
            "lag_4": l4,
            "roll_4_mean": (l1 + l2 + l3 + l4) / 4,
            "cell_lat": wlat,
            "cell_lon": wlon
        }])

        res = predict(
            row,
            reg,
            clf,
            features
        ).iloc[0]

        # PREDICTION KPIs

        m1, m2, m3 = st.columns(3)

        m1.metric(
            "Predicted Crimes Next Week",
            f"{res['predicted_crimes']:.1f}"
        )

        m2.metric(
            "Hotspot Probability",
            f"{res['hotspot_probability'] * 100:.0f}%"
        )

        m3.metric(
            "Risk Level",
            res["risk_level"]
        )

        # CLASSIFIER-BASED HOTSPOT DECISION

        st.markdown("---")

        if res["hotspot_probability"] >= 0.5:

            st.error(
                "🔥 This area is classified as a "
                "LIKELY HOTSPOT by the classification model."
            )

        else:

            st.success(
                "✅ This area is classified as "
                "NOT A HOTSPOT by the classification model."
            )

        st.caption(
            f"The classification model uses the training-defined "
            f"hotspot rule: more than {threshold:g} crimes per week. "
            "The decision above uses a 0.5 probability cutoff."
        )


# --------------------------------------------------
# 7. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "⚠️ Educational project. Predictions are based on reported "
    "Chicago crime data (2022). They describe locations, "
    "not people, and do not guarantee any outcome."
)
