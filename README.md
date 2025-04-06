# Sales-Prediction-using-Time-Series-Analysis
Sales prediction using Time Series Analysis
# 📊 Time Series Analysis & Forecasting with ARIMA and Prophet

This repository contains an end-to-end time series analysis pipeline including decomposition, stationarity testing, ARIMA modeling, and Prophet-based forecasting. It leverages various visualization tools such as Plotly and Matplotlib to understand seasonal trends and forecast future values.

---

## 📁 Dataset

- **File:** `Month_Value_1.csv`  
- **Target:** `Revenue`, `Sales_quantity`, `Average_cost`, `The_average_annual_payroll_of_the_region`  
- **Date Column:** `Date`  
- **Objective:** Analyze and forecast trends using time series modeling techniques.

---

## 🔍 Project Breakdown

### 1. Load & Prepare Data

```python
import pandas as pd

df = pd.read_csv("Month_Value_1.csv")
df.rename(columns={'Period': 'Date'}, inplace=True)
df['Date'] = pd.to_datetime(df['Date'])
```

---

### 2. Data Visualization & Exploration

#### 📈 Line Plot with Plotly

```python
import plotly.express as px

df['Month_Year'] = df['Date'].dt.to_period('M').dt.to_timestamp()
df_melted = df.melt(id_vars=['Month_Year'], value_vars=df.columns[1:-1],
                    var_name='Category', value_name='Value')

fig = px.line(df_melted, x='Month_Year', y='Value', color='Category',
              title='Monthly Trend by Category')
fig.show()
```

#### 📊 Stacked Bar Plot

```python
fig = px.bar(df, x='Month_Year', y=df.columns[1:], barmode='stack')
fig.show()
```

#### 📉 Scatter Plot with Trendline

```python
fig = px.scatter(df, x='Sales_quantity', y='Average_cost',
                 color='Year', trendline='ols', marginal_x='box', marginal_y='box')
fig.show()
```

---

### 3. Imputation & Correlation Analysis

```python
from sklearn.impute import KNNImputer

imputer = KNNImputer()
df[columns_to_impute] = imputer.fit_transform(df[columns_to_impute])
```

#### 🔥 Correlation Heatmap

```python
import seaborn as sns
import matplotlib.pyplot as plt

corr = df.corr()
sns.heatmap(corr, annot=True)
plt.show()
```

---

### 4. Stationarity & Decomposition

```python
from statsmodels.tsa.stattools import adfuller
result = adfuller(df['Average_cost'])
```

#### 🔄 Seasonal Decomposition

```python
from statsmodels.tsa.seasonal import seasonal_decompose

decompose = seasonal_decompose(df['Revenue'], model='additive', period=30)
decompose.plot()
```

---

### 5. ARIMA Forecasting

```python
from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(df['Sales_quantity'], order=(1,1,1))
model_fit = model.fit()
model_fit.summary()
```

---

### 6. Prophet Forecasting

```python
from prophet import Prophet

df = df.rename(columns={'Date': 'ds', 'Revenue': 'y'})
model = Prophet()
model.fit(df)

future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)
```

---

## 🛠 Tools Used

- Python (Pandas, Scikit-learn, Prophet, Statsmodels)
- Visualization: Plotly, Seaborn, Matplotlib
- Forecasting Models: ARIMA, Prophet
- Feature Engineering: Differencing, Lag Features

---

## 🚀 How to Run

1. Clone this repository  
2. Place `Month_Value_1.csv` in the root folder  
3. Install requirements:  
   ```bash
   pip install pandas numpy plotly seaborn matplotlib statsmodels prophet
   ```
4. Run the notebook or script to view results and plots  
