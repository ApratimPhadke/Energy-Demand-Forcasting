# ⚡ LSTM-Based Time Series Forecasting on Household Power Consumption

> **An end-to-end deep learning project** leveraging LSTM networks for energy demand forecasting, complete with statistical validation, exploratory data analysis (EDA), feature engineering, and explainable AI (SHAP) interpretation.

---

## 🧭 Table of Contents

* [Overview](#-overview)
* [Dataset Description](#-dataset-description)
* [Key Objectives](#-key-objectives)
* [Project Pipeline](#-project-pipeline)
* [Feature Engineering](#-feature-engineering)
* [Exploratory Data Analysis (EDA)](#-exploratory-data-analysis-eda)
* [Model Architecture](#-model-architecture)
* [Evaluation Metrics](#-evaluation-metrics)
* [Model Explainability (SHAP)](#-model-explainability-shap)
* [Results & Insights](#-results--insights)
* [Installation & Execution](#-installation--execution)
* [Project Structure](#-project-structure)
* [Future Work](#-future-work)
* [References](#-references)

---

## 📘 Overview

This project demonstrates **time series modeling and forecasting** using **Long Short-Term Memory (LSTM)** neural networks on the *Household Power Consumption* dataset.
The pipeline includes:

* Statistical testing (ADF Test, Normality Check)
* Multi-level time aggregation (Daily, Weekly, Monthly)
* Deep Learning Forecasting (LSTM)
* Explainability via **SHAP** (Global & Local Interpretations)

The project bridges **classical statistical analysis** and **modern neural forecasting** to uncover deep insights into household energy patterns.

---

## 📊 Dataset Description

**Dataset Name:** `household_power_consumption.csv`
**Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption)

| Column                | Description                                                                  |
| :-------------------- | :--------------------------------------------------------------------------- |
| `Date`, `Time`        | Timestamp of power readings                                                  |
| `Global_active_power` | Total household active power (kW)                                            |
| Other columns         | Include voltage, current, sub-metering, etc. (optional fields not used here) |

* **Time Period Covered:** Dec 2006 – Nov 2010
* **Sampling Frequency:** 1 minute
* **Data Size:** 2,075,259 rows × 9 columns

---

## 🎯 Key Objectives

1. **Understand** the underlying power consumption patterns.
2. **Preprocess & engineer** temporal and statistical features.
3. **Build and optimize** an LSTM model for sequence prediction.
4. **Quantify performance** using RMSE and MAE metrics.
5. **Interpret model behavior** using SHAP-based explainability.

---

## 🧩 Project Pipeline

```
Data Loading → Cleaning → Feature Engineering →
EDA → Stationarity Testing → LSTM Modeling →
Performance Evaluation → Explainability (SHAP)
```

---

## ⚙️ Feature Engineering

Key transformations applied:

| Step               | Transformation                                         |
| ------------------ | ------------------------------------------------------ |
| Timestamp parsing  | Combined `Date` + `Time` → `date_time`                 |
| Numerical cleaning | Converted `Global_active_power` → float; dropped NaNs  |
| Derived features   | Extracted `year`, `quarter`, `month`, `day`, `weekday` |
| Sorting & indexing | Chronologically ordered time series                    |
| Scaling            | `MinMaxScaler(0,1)` for LSTM normalization             |

---

## 🔍 Exploratory Data Analysis (EDA)

**1. Normality Test (D’Agostino’s K²):**

```python
stat, p = stats.normaltest(data.Global_active_power)
```

Result: **Non-Gaussian distribution** → Heavy skew and kurtosis observed.

**2. Visual Insights:**

* Time series plots (2006–2008)
* Violin plots across Year & Quarter
* Histograms & Probability plots
* Aggregated daily, weekly, monthly, quarterly, yearly power consumption trends

**3. Stationarity Testing:**
Augmented Dickey-Fuller test confirms **non-stationarity**, justifying LSTM usage.

---

## 🧠 Model Architecture

**Model Type:** LSTM (Sequential)

| Layer | Type             | Parameters                        |
| :---- | :--------------- | :-------------------------------- |
| 1     | LSTM (100 units) | Input: (timesteps=30, features=1) |
| 2     | Dropout(0.2)     | Regularization                    |
| 3     | Dense(1)         | Output layer                      |

**Compilation:**

```python
model.compile(loss='mean_squared_error', optimizer='adam')
```

**Training:**

* `epochs=20`
* `batch_size=1240`
* `validation_split=0.2`
* `EarlyStopping(monitor='val_loss', patience=4)`

---

## 📈 Evaluation Metrics

| Metric       | Formula                        | Purpose                      |   |                                   |
| :----------- | :----------------------------- | :--------------------------- | - | --------------------------------- |
| **MAE**      | mean(                          | y_true - y_pred              | ) | Average absolute prediction error |
| **RMSE**     | sqrt(mean((y_true - y_pred)²)) | Penalizes larger errors more |   |                                   |
| **R² Score** | Model variance explanation     | Optional (not used here)     |   |                                   |

### Example Output

```
Train MAE: 0.019
Train RMSE: 0.027
Test MAE: 0.021
Test RMSE: 0.029
```

---

## 🧩 Model Explainability (SHAP)

Explainability is implemented using both **DeepExplainer** and **KernelExplainer** from the SHAP library.

### Global Feature Importance

`shap.summary_plot()` visualizes the most influential time lags (`t-30` → `t-1`) driving the prediction.

### Local Explanation

`shap.force_plot()` provides **instance-level interpretation** — showing which recent values push predictions higher or lower.

These plots transform the model into an **interpretable forecasting tool**, crucial for industrial applications.

---

## 📊 Results & Insights

✅ The LSTM model successfully captured temporal dependencies in power usage.
✅ Strong performance on test data with low RMSE.
✅ SHAP analysis revealed that **recent 7–10 timesteps** most influence next-step predictions.
✅ Clear seasonal & diurnal patterns were observed in EDA visualizations.

---

## 🧠 Future Work

* Implement **Bidirectional LSTM** and **GRU** for comparison.
* Add **external regressors** (temperature, holidays).
* Deploy the model as a **Flask/Streamlit dashboard**.
* Integrate **real-time prediction pipelines** via MQTT or Kafka.

---

## 🧰 Installation & Execution

### Prerequisites

* Python ≥ 3.8
* Libraries:

  ```bash
  pip install numpy pandas matplotlib seaborn scikit-learn keras tensorflow shap statsmodels
  ```

### Run the Project

```bash
python untitled3.py
```

or in Jupyter/Colab:

```python
!python untitled3.py
```

---

## 🗂️ Project Structure

```
├── household_power_consumption.csv   # Dataset
├── untitled3.py                      # Main project file
├── README.md                         # Project documentation
└── requirements.txt                  # Dependencies (optional)
```

---

## 📚 References

* UCI Machine Learning Repository — Household Power Consumption Dataset
* Chollet, F. (2015). *Keras: Deep Learning Library for Theano and TensorFlow*
* Lundberg, S. & Lee, S. (2017). *A Unified Approach to Interpreting Model Predictions (SHAP)*
* Hyndman, R. & Athanasopoulos, G. (2018). *Forecasting: Principles and Practice*

---

## 🏁 Author

**Apratim Phadke** and **Ishika Bhad**
📧 [GitHub](https://github.com/ApratimPhadke) | 🔗 [LinkedIn-Apratim Phadke](https://www.linkedin.com/in/apratim-phadke-966816223/)|
🔗 [LinkedIn-Ishika Bhad](https://www.linkedin.com/in/ishika-bhad-a47ab0295/)

