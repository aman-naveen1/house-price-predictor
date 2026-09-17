# 🏠 House Price Predictor

A beginner-friendly end-to-end machine learning project that predicts California house values using regression models. The project covers data loading, exploratory data analysis, preprocessing, model comparison, evaluation, feature importance, and an interactive Streamlit app.

## 🚀 What this project demonstrates

- Python + Pandas data analysis
- Exploratory data analysis with Matplotlib and Seaborn
- Train/test splitting
- Data preprocessing with scikit-learn pipelines
- Linear Regression vs Random Forest Regression
- MAE, RMSE and R² evaluation
- Feature importance analysis
- Interactive prediction UI with Streamlit
- Reproducible project structure

## 📊 Dataset

This project uses the **California Housing** dataset provided through `sklearn.datasets.fetch_california_housing`.

The target is median house value, represented by scikit-learn in units of **$100,000**. For example, a model output of `2.50` corresponds to approximately `$250,000`.

The eight input features are:

| Feature | Meaning |
|---|---|
| MedInc | Median income in the block group |
| HouseAge | Median house age |
| AveRooms | Average number of rooms |
| AveBedrms | Average number of bedrooms |
| Population | Block-group population |
| AveOccup | Average household occupancy |
| Latitude | Geographic latitude |
| Longitude | Geographic longitude |

## 🧠 Project workflow

```text
Dataset
   ↓
Exploratory Data Analysis
   ↓
Train / Test Split
   ↓
Preprocessing Pipeline
   ↓
Linear Regression + Random Forest
   ↓
MAE / RMSE / R² comparison
   ↓
Feature Importance
   ↓
Interactive Streamlit App
```

## 📁 Project structure

```text
house-price-predictor/
├── README.md
├── train.py
├── app.py
├── house_price_predictor.ipynb
├── requirements.txt
├── .gitignore
└── data/
    └── README.md
```

## ⚙️ Setup

### 1. Clone the repository

```bash
git clone https://github.com/aman-naveen1/house-price-predictor.git
cd house-price-predictor
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 🧪 Train and evaluate the models

```bash
python train.py
```

This downloads the dataset through scikit-learn, trains both models, prints MAE/RMSE/R², and creates evaluation plots in `artifacts/`.

## 🌐 Run the web app

```bash
streamlit run app.py
```

The app lets you enter the eight housing features and returns an estimated median house value.

## 📈 Evaluation metrics

The training script reports:

- **MAE (Mean Absolute Error):** average absolute prediction error.
- **RMSE (Root Mean Squared Error):** penalizes larger errors more strongly.
- **R²:** proportion of target variance explained by the model.

Metrics are intentionally generated when the project is run rather than hard-coded, so the repository does not claim fabricated results.

## 💡 Why Random Forest?

Linear Regression provides a simple baseline. Random Forest can model nonlinear relationships and interactions between features, making it a useful second model for comparison.

## ⚠️ Limitations

This is an educational machine-learning project, not a professional property valuation system. The model is trained on California data and should not be used to estimate real-world property prices outside the dataset's distribution.

## 🧾 Resume bullet

> Built an end-to-end house price regression project using Python, Pandas and scikit-learn; compared Linear Regression and Random Forest using MAE, RMSE and R², analyzed feature importance, and deployed an interactive Streamlit prediction app.

## 📌 Future improvements

- Add cross-validation and hyperparameter tuning
- Add a larger or more recent housing dataset
- Add model explainability with SHAP
- Deploy the Streamlit application publicly
- Add automated tests and CI
