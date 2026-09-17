"""Train and evaluate house-price regression models.

Run with:
    python train.py

The script downloads the California Housing dataset through scikit-learn,
trains Linear Regression and Random Forest models, prints evaluation metrics,
and saves plots under artifacts/.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
ARTIFACTS = Path("artifacts")


def load_data():
    """Load the California Housing dataset as a pandas DataFrame."""
    housing = fetch_california_housing(as_frame=True)
    return housing.frame, housing.feature_names


def evaluate(model, X_test, y_test):
    """Return standard regression metrics."""
    predictions = model.predict(X_test)
    return {
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": mean_squared_error(y_test, predictions) ** 0.5,
        "R2": r2_score(y_test, predictions),
    }, predictions


def main():
    ARTIFACTS.mkdir(exist_ok=True)

    df, feature_names = load_data()
    X = df[feature_names]
    y = df["MedHouseVal"]

    print(f"Dataset shape: {df.shape}")
    print("\nMissing values:")
    print(X.isna().sum())

    # Quick EDA plot: target distribution.
    plt.figure(figsize=(9, 5))
    sns.histplot(y, bins=40, kde=True)
    plt.title("Distribution of Median House Value")
    plt.xlabel("Median house value ($100,000 units)")
    plt.ylabel("Number of observations")
    plt.tight_layout()
    plt.savefig(ARTIFACTS / "target_distribution.png", dpi=160)
    plt.close()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )

    # Linear regression benefits from standardization.
    linear_model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )

    # Random Forest does not require feature scaling, but an imputer keeps
    # the pipeline robust if missing values appear in future data.
    forest_model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=250,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    models = {
        "Linear Regression": linear_model,
        "Random Forest": forest_model,
    }

    results = []
    predictions_by_model = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        metrics, predictions = evaluate(model, X_test, y_test)
        results.append({"Model": name, **metrics})
        predictions_by_model[name] = predictions

    results_df = pd.DataFrame(results).sort_values("R2", ascending=False)

    print("\nModel comparison:")
    print(results_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    # Compare actual vs predicted values for both models.
    for name, predictions in predictions_by_model.items():
        plt.figure(figsize=(7, 6))
        plt.scatter(y_test, predictions, alpha=0.35)
        lower = min(y_test.min(), predictions.min())
        upper = max(y_test.max(), predictions.max())
        plt.plot([lower, upper], [lower, upper], linestyle="--")
        plt.title(f"Actual vs Predicted — {name}")
        plt.xlabel("Actual value ($100,000 units)")
        plt.ylabel("Predicted value ($100,000 units)")
        plt.tight_layout()
        safe_name = name.lower().replace(" ", "_")
        plt.savefig(ARTIFACTS / f"actual_vs_predicted_{safe_name}.png", dpi=160)
        plt.close()

    # Feature importance from the fitted Random Forest.
    forest = forest_model.named_steps["model"]
    importances = pd.Series(forest.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=False)

    plt.figure(figsize=(9, 5))
    sns.barplot(x=importances.values, y=importances.index)
    plt.title("Random Forest Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(ARTIFACTS / "feature_importance.png", dpi=160)
    plt.close()

    results_df.to_csv(ARTIFACTS / "model_metrics.csv", index=False)

    print("\nTop Random Forest features:")
    print(importances.head(5).to_string())
    print(f"\nSaved evaluation artifacts to: {ARTIFACTS.resolve()}")


if __name__ == "__main__":
    main()
