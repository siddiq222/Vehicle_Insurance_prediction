# Health Insurance Cross-Sell Prediction using ANN
## Vehicle_Insurance_prediction

Predicting whether an existing health insurance customer will be interested in purchasing vehicle insurance, using an Artificial Neural Network built with TensorFlow/Keras — deployed as an interactive Streamlit app.


---

## 📌 Problem Statement

An insurance company wants to identify which of its existing health insurance customers are likely to be interested in also purchasing vehicle insurance (a cross-sell opportunity). Being able to predict this in advance allows the company to run a targeted, efficient marketing campaign instead of contacting its entire customer base.

## 🎯 Objective

Build a binary classification model using an Artificial Neural Network that predicts customer interest (`Response`: 1 = Yes, 0 = No), based on demographic, vehicle, and policy-related features — and deploy it as a usable, real-time prediction app.

---

## 🗂️ Dataset

- **Source**: [Health Insurance Cross Sell Prediction (Kaggle)](https://www.kaggle.com/datasets/anmolkumar/health-insurance-cross-sell-prediction)
- **Size**: 381,109 rows × 12 columns
- **Target**: `Response` — significantly imbalanced (~87.5% No / ~12.5% Yes)

| Feature | Description |
|---|---|
| Gender, Age | Customer demographics |
| Driving_License | 0/1 |
| Region_Code | Encoded region |
| Previously_Insured | 0/1 — already has vehicle insurance |
| Vehicle_Age | <1 Year, 1-2 Year, >2 Years |
| Vehicle_Damage | Yes/No |
| Annual_Premium | Premium amount |
| Policy_Sales_Channel | Encoded outreach channel |
| Vintage | Days associated with the company |
| Response | Target variable |

---

## 🔍 Key EDA Findings

- **Previously_Insured** and **Vehicle_Damage** are the strongest individual predictors — customers not previously insured *and* with past vehicle damage respond at **~25.5%**, more than double the baseline rate.
- **Policy_Sales_Channel** shows a ~10x variation in response rate across channels.
- Chi-square testing confirmed a strong statistical association between Previously_Insured and Vehicle_Damage (**Cramér's V = 0.82**).
- Annual_Premium, Vintage, and Driving_License showed little to no predictive signal.

Full analysis (univariate, bivariate, multivariate, statistical tests) is in [`notebooks/ANN_Insurance_Project.ipynb`](notebooks/).

---

## ⚙️ Preprocessing Pipeline

1. Stratified train-test split (80/20) — performed **before** any statistic-based preprocessing to avoid data leakage.
2. Ordinal encoding for `Vehicle_Age`.
3. Frequency encoding for high-cardinality `Region_Code` and `Policy_Sales_Channel` (fit on train, applied to test).
4. Log transformation (`log1p`) on `Annual_Premium` to correct heavy right-skew.
5. One-hot encoding for `Gender` and `Vehicle_Damage`.
6. `StandardScaler` applied to all numerical features.

All fitted preprocessing objects (scaler, encoders, frequency maps) are saved in [`preprocessing/`](preprocessing/) for consistent reuse at inference time.

---

## 🧠 Model

**Architecture**
```
Dense(64, relu) → Dropout(0.1)
Dense(24, relu) → Dropout(0.3)
Dense(1, sigmoid)
```

- **Loss**: Binary Crossentropy
- **Optimizer**: Adam (learning rate: 0.0005)
- **Class imbalance handling**: class weights (`{0: 0.57, 1: 4.08}`) computed via `sklearn.compute_class_weight`
- **Hyperparameter tuning**: [Optuna](https://optuna.org/) — 20 trials, Bayesian/TPE search


---

## 🚀 Streamlit App

An interactive app for real-time prediction on new customer data.

**Run locally:**
```bash
git clone https://github.com/siddiq222/Vehicle_Insurance_prediction
cd Vehicle_Insurance_prediction
## creating environment
python -m venv yourenv
## activating environment
.\yourenv\Scripts\activate
## installing required libraries
pip install -r requirements.txt
## executing
streamlit run app.py

##deactivate the environment
deactivate
##delete the environment if you want
rmdir /s /q yourenv
```

**App pages:**
- **Home** — project overview
- **EDA** — key visualizations and insights
- **Make Predictions** — enter customer details and get a real-time prediction with confidence score

---

## 🛠️ Tech Stack

- **Language**: Python 3.10
- **Data & Modeling**: Pandas, NumPy, Scikit-learn, TensorFlow/Keras
- **Visualization**: Matplotlib, Seaborn
- **Statistical Testing**: SciPy
- **Hyperparameter Tuning**: Optuna
- **Deployment**: Streamlit
- **Model Persistence**: Joblib, Keras `.keras` format

---

## 📁 Project Structure

```
├── run.py                          # Streamlit app entry point
├── module.py / run.py / predictions.py  # Streamlit page modules
├── notebooks/
│   └── ANN_Insurance_Project.ipynb # Full EDA + modeling notebook
├── preprocessing/
│   ├── scaler.pkl
│   ├── region_freq.pkl
│   ├── channel_freq.pkl
│   └── onehot_encoder.pkl
├── final_ann_model.keras           # Trained model
├── requirements.txt
└── README.md
```

---

## 🔮 Future Work

- Threshold tuning / precision-recall curve to let stakeholders pick a custom operating point
- SHAP / permutation feature importance to validate EDA findings against learned model behavior
- Target encoding for `Policy_Sales_Channel` as an alternative to frequency encoding
- Benchmark against gradient-boosted tree models (XGBoost, LightGBM)

---

## 📄 License

This project is for educational purposes as part of a personal/academic data science project.