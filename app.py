"""Interactive Streamlit house-price prediction app.

Run with:
    streamlit run app.py
"""

import streamlit as st
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

FEATURES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]


@st.cache_resource
def get_model():
    """Download the dataset once and train the Random Forest model."""
    housing = fetch_california_housing(as_frame=True)
    X = housing.data
    y = housing.target

    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=250,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(X, y)
    return model


st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered",
)

st.title("🏠 California House Price Predictor")
st.write(
    "Enter the characteristics of a California block group and the model "
    "will estimate its median house value."
)

st.info(
    "The model is trained on the scikit-learn California Housing dataset. "
    "The target is expressed in $100,000 units."
)

with st.form("prediction_form"):
    st.subheader("Property & location details")

    col1, col2 = st.columns(2)

    with col1:
        med_inc = st.number_input(
            "Median income (MedInc)", min_value=0.0, max_value=20.0,
            value=3.5, step=0.1,
            help="Median income in the block group, measured in tens of thousands of dollars.",
        )
        house_age = st.number_input(
            "House age (years)", min_value=1.0, max_value=60.0,
            value=25.0, step=1.0,
        )
        avg_rooms = st.number_input(
            "Average rooms", min_value=1.0, max_value=20.0,
            value=5.0, step=0.1,
        )
        avg_bedrooms = st.number_input(
            "Average bedrooms", min_value=0.2, max_value=10.0,
            value=1.0, step=0.1,
        )

    with col2:
        population = st.number_input(
            "Population", min_value=1.0, max_value=50000.0,
            value=1000.0, step=100.0,
        )
        avg_occupancy = st.number_input(
            "Average occupancy", min_value=0.5, max_value=20.0,
            value=3.0, step=0.1,
        )
        latitude = st.number_input(
            "Latitude", min_value=32.0, max_value=42.5,
            value=35.5, step=0.1,
        )
        longitude = st.number_input(
            "Longitude", min_value=-125.0, max_value=-114.0,
            value=-119.5, step=0.1,
        )

    submitted = st.form_submit_button("Predict House Value")

if submitted:
    model = get_model()
    values = [[
        med_inc,
        house_age,
        avg_rooms,
        avg_bedrooms,
        population,
        avg_occupancy,
        latitude,
        longitude,
    ]]

    prediction_100k = float(model.predict(values)[0])
    prediction_dollars = prediction_100k * 100_000

    st.success(f"Estimated median house value: ${prediction_dollars:,.0f}")
    st.caption(
        f"Raw model output: {prediction_100k:.3f} × $100,000. "
        "This is an educational prediction, not a professional valuation."
    )

st.divider()
st.caption("Built with Python, Pandas, scikit-learn and Streamlit.")
