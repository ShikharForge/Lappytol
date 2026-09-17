# 💻 Laptop Price Prediction

An end-to-end machine learning project that predicts laptop prices (INR) from hardware specifications. Built with scikit-learn, XGBoost, SHAP, and Streamlit.

---

## Problem Statement

Laptop prices depend on a complex interaction of brand, processor tier, GPU class, memory, display quality, and form factor. This project builds a regression model that learns these relationships from 1,303 real laptop listings and produces price estimates in INR. The goal is a reproducible pipeline — from raw data to interpretable model to working web application — without inflating metrics or overstating scope.

---

## Dataset

| Attribute | Detail |
|---|---|
| File | `laptop data/laptop_data.csv` |
| Rows | 1,303 |
| Original columns | 12 |
| Target | `Price` (INR) |
| Missing values | None |
| Duplicate rows | None |
| Price range | approx. ₹9,271 – ₹3,24,954 |
| Price skewness (raw) | ≈ 1.52 (right-skewed) |

**Original columns:** `Company`, `TypeName`, `Inches`, `ScreenResolution`, `Cpu`, `Ram`, `Memory`, `Gpu`, `OpSys`, `Weight`, `Price`

---

## Project Workflow

```
Phase 0 — Dataset Inspection
Phase 1 — Data Cleaning, Feature Engineering, EDA
Phase 2 — Model Development & Evaluation
Phase 3 — Final Model, SHAP Interpretation, Error Analysis, Prediction Pipeline
Phase 4 — Streamlit Application, GitHub Polish
```

---

## Feature Engineering

All features are derived from the raw CSV columns. No external data is used.

| Group | Engineered Features |
|---|---|
| RAM | `ram_gb` |
| Weight | `weight_kg` |
| Screen | `resolution_width`, `resolution_height`, `total_pixels`, `is_touchscreen`, `is_ips` |
| CPU | `cpu_brand`, `cpu_family` (ordinal tier), `cpu_speed_ghz` |
| GPU | `gpu_brand`, `gpu_family` (ordinal tier) |
| Storage | `ssd_gb`, `hdd_gb`, `flash_storage_gb`, `hybrid_gb`, `total_storage_gb`, `primary_storage_type` |

**CPU family tiers** (low → high): Atom < Celeron < Pentium < Core i3 < Core M < Ryzen < Core i5 < Core i7 < Xeon

**GPU family tiers** (low → high): Intel HD < Intel UHD < Intel Iris < GeForce < GeForce GTX < GeForce RTX < Quadro

---

## Exploratory Analysis

See [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) for the full EDA, including:

- Price distribution (raw and log-transformed)
- Price vs RAM, CPU family, GPU, storage type
- Correlation heatmap
- Category frequency charts
- Outlier analysis

**Key observations:**
- RAM is the strongest correlated numerical feature with price
- CPU and GPU tier show clear monotonic price gradients
- SSDs command a significant premium over HDDs of equal capacity
- IPS and touchscreen displays add meaningful price premiums
- 19 companies present; Apple, Lenovo, Dell, HP, Asus account for most listings

<!-- TODO: Add EDA screenshot here -->
<!-- ![EDA visualization](assets/eda_price_distribution.png) -->

---

## Models

Six model families were trained with both raw-price and log-price targets, evaluated using 5-fold cross-validation on the training set. All metrics are in INR (log-price models are inverse-transformed with `expm1` before scoring).

### Model Comparison

| Model | Target | CV R² | CV MAE | CV RMSE |
|---|---|---|---|---|
| **XGBoost (Tuned)** | **Log** | **0.8364** | **₹9,273** | **₹14,916** |
| XGBoost | Raw | 0.8311 | ₹9,797 | ₹15,094 |
| XGBoost | Log | 0.8280 | ₹9,528 | ₹15,259 |
| GradientBoosting | Log | 0.8251 | ₹9,860 | ₹15,400 |
| GradientBoosting | Raw | 0.8224 | ₹10,329 | ₹15,519 |
| RandomForest | Raw | 0.8112 | ₹10,044 | ₹15,964 |
| RandomForest (Tuned) | Log | 0.8039 | ₹9,935 | ₹16,291 |
| LinearRegression | Raw | 0.7712 | ₹12,358 | ₹17,527 |
| Ridge | Raw | 0.7709 | ₹12,505 | ₹17,583 |
| DummyRegressor | — | −0.003 | ₹28,835 | ₹37,000 |

See [`notebooks/02_modeling.ipynb`](notebooks/02_modeling.ipynb) for the full comparison, tuning runs, and residual diagnostics.

<!-- TODO: Add model comparison chart here -->
<!-- ![Model comparison](assets/model_comparison.png) -->

---

## Final Model

**XGBoost Regressor with log1p(Price) target**

### Why log1p?

Raw prices are right-skewed (skewness ≈ 1.52). Predicting `log(Price + 1)` instead of raw Price:
- Compresses the extreme high-price range
- Produces a more symmetric target distribution
- Improves model fit across the full price range

Predictions are inverse-transformed back to INR using `numpy.expm1()`. All reported metrics are in INR.

### Hyperparameters (from RandomizedSearchCV, 30 iterations, 5-fold CV)

| Parameter | Value |
|---|---|
| `n_estimators` | 600 |
| `max_depth` | 4 |
| `learning_rate` | 0.05 |
| `subsample` | 0.7 |
| `colsample_bytree` | 1.0 |

### Verified Metrics

| Split | R² | MAE | RMSE |
|---|---|---|---|
| 5-fold CV (train) | 0.8364 | ₹9,273 | ₹14,916 |
| Held-out test (20%) | **0.8504** | **₹8,253** | **₹14,686** |

The test set was held out completely during all preprocessing fitting, model training, and hyperparameter tuning.

---

## Error Analysis

See [`notebooks/03_model_interpretation.ipynb`](notebooks/03_model_interpretation.ipynb) for the full error analysis.

**By price segment** (bands from training-set quartiles):

| Segment | Price Band | Test MAE |
|---|---|---|
| Budget | Bottom 25% | Lower |
| Mid-Low | 25th–50th percentile | Moderate |
| Mid-High | 50th–75th percentile | Moderate |
| Premium | Top 25% | Higher |

Premium laptops have higher absolute errors because the log transform compresses the upper end, and fewer premium examples exist in training data.

---

## SHAP Interpretation

The model is explained using SHAP (SHapley Additive exPlanations). SHAP values are in **log-price units**. A SHAP value of +0.5 means a feature increased the log-price prediction by 0.5, approximately a 1.65× multiplicative effect on the final price.

**Top drivers of price (in order of mean |SHAP|):**
1. `ram_gb`
2. `cpu_family` (ordinal tier)
3. `gpu_family` / `gpu_brand`
4. `total_pixels`
5. `primary_storage_type` / `ssd_gb`

<!-- TODO: Add SHAP summary plot here -->
<!-- ![SHAP summary](assets/shap_summary.png) -->

---

## Streamlit Application

A local web application for interactive price prediction.

### Screenshot

<!-- TODO: Add app screenshot here -->
<!-- ![Streamlit app](assets/app_screenshot.png) -->

### Running the app

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model (if models/ artifacts are missing)
python src/train_final.py

# Run the app
streamlit run app.py
```

The app opens at `http://localhost:8501`.

### Usage

1. Select laptop specifications using the dropdowns and slider
2. Click **Predict Price**
3. The app displays the estimated price in INR, your entered specifications, and the log-price intermediate value

---

## Installation

```bash
# Clone the repository
git clone https://github.com/ShikharForge/Lappytol.git
cd Lappytol

# Install dependencies
pip install -r requirements.txt

# Train the model
python src/train_final.py
```

---

## Usage

### Prediction from Python

```python
from src.predict import predict_price

result = predict_price({
    "Company":          "Dell",
    "TypeName":         "Notebook",
    "Inches":           15.6,
    "ScreenResolution": "Full HD 1920x1080",
    "Cpu":              "Intel Core i5 7200U 2.5GHz",
    "Ram":              "8GB",
    "Memory":           "256GB SSD",
    "Gpu":              "Intel HD Graphics 620",
    "OpSys":            "Windows 10",
    "Weight":           "1.86kg",
})
# Returns: 47216.23  (float, INR)
```

### Retrain the model

```bash
python src/train_final.py
```

### Run notebooks

```bash
jupyter notebook notebooks/
```

---

## Limitations

1. **Premium segment accuracy:** Laptops priced above ≈₹2,00,000 are underrepresented in the training data, leading to higher prediction errors.
2. **No temporal dimension:** The dataset is a static snapshot. Real-world prices change; the model does not capture time-based trends.
3. **Rule-based parsing:** CPU, GPU, and storage features are extracted using string matching. Atypical naming conventions may produce incorrect feature values.
4. **Rare brands:** Brands with very few training examples (e.g., Vero, Chuwi, Mediacom) may have higher prediction variance.
5. **Currency and region:** All prices are in INR from a specific time period and source. The model is not suitable for other currencies or markets without retraining.
6. **No confidence intervals:** The model produces point estimates. Uncertainty quantification is not implemented.

---

## Future Improvements

- Add temporal price data to capture market trends
- Improve GPU tier extraction to cover newer GPU generations (RTX 4000 series)
- Train on a larger, more recent dataset
- Add proper prediction intervals using quantile regression or conformal prediction
- Experiment with stacking or blending top models
- Fine-grained CPU identification (TDP, core count) for better power segment separation

---

## Project Structure

```
├── laptop data/
│   └── laptop_data.csv             # Original dataset (never modified)
├── notebooks/
│   ├── 01_eda.ipynb                # Exploratory data analysis
│   ├── 02_modeling.ipynb           # Model training and comparison
│   └── 03_model_interpretation.ipynb  # SHAP + error analysis
├── src/
│   ├── preprocessing.py            # Data cleaning pipeline
│   ├── features.py                 # Feature engineering
│   ├── train_final.py              # Trains and saves final model
│   └── predict.py                  # Reusable predict_price() function
├── models/
│   ├── final_pipeline.joblib       # Fitted sklearn Pipeline
│   ├── feature_metadata.joblib     # Feature metadata and metrics
│   └── train_test_split.joblib     # Saved train/test split
├── app.py                          # Streamlit application
├── requirements.txt
├── .gitignore
└── README.md
```
