import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude",
]


def make_toy_data(n=60):
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(n, len(FEATURES))), columns=FEATURES)
    y = 2.0 + 0.8 * X["MedInc"] - 0.3 * X["HouseAge"] + rng.normal(0, 0.1, n)
    X.loc[0, "MedInc"] = np.nan
    return X, y


def test_linear_pipeline_predicts():
    X, y = make_toy_data()
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LinearRegression()),
    ])
    model.fit(X, y)
    predictions = model.predict(X.iloc[:5])
    assert len(predictions) == 5
    assert np.isfinite(predictions).all()


def test_random_forest_pipeline_predicts():
    X, y = make_toy_data()
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(n_estimators=10, random_state=42, n_jobs=-1)),
    ])
    model.fit(X, y)
    predictions = model.predict(X.iloc[:5])
    assert len(predictions) == 5
    assert np.isfinite(predictions).all()
