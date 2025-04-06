# Time Series Analysis and Forecasting

This project is a complete pipeline for time series analysis, modeling, and forecasting using ARIMA and Prophet. It also includes visualizations for data understanding and trend analysis.

---

## Dataset

- **File:** `Month_Value_1.csv`
- **Target Columns:** `Revenue`, `Sales_quantity`, `Average_cost`, `The_average_annual_payroll_of_the_region`
- **Date Column:** `Date`
- **Goal:** Analyze trends and forecast future values for key metrics.

---

## Project Structure

### 1. Load and Prepare Data

```python
import pandas as pd

df = pd.read_csv("Month_Value_1.csv")
df.rename(columns={'Period': 'Date'}, inplace=True)
df['Date'] = pd.to_datetime(df['Date'])
```

---

### 2. Data Visualization and Exploration

#### Line Plot

```python
import plotly.express as px

df['Month_Year'] = df['Date'].dt.to_period('M').dt.to_timestamp()
df_melted = df.melt(id_vars=['Month_Year'], value_vars=df.columns[1:-1],
                    var_name='Category', value_name='Value')
fig = px.line(df_melted, x='Month_Year', y='Value', color='Category')
fig.show()
```

#### Stacked Bar Plot

```python
fig = px.bar(df, x='Month_Year', y=df.columns[1:], barmode='stack')
fig.show()
```

#### Scatter Plot with Trendline

```python
fig = px.scatter(df, x='Sales_quantity', y='Average_cost',
                 color='Year', trendline='ols', marginal_x='box', marginal_y='box')
fig.show()
```

---

### 3. Imputation and Correlation

```python
from sklearn.impute import KNNImputer

imputer = KNNImputer()
df[columns_to_impute] = imputer.fit_transform(df[columns_to_impute])
```

#### Correlation Heatmap

```python
import seaborn as sns
import matplotlib.pyplot as plt

corr = df.corr()
sns.heatmap(corr, annot=True)
plt.show()
```

---

### 4. Stationarity and Decomposition

```python
from statsmodels.tsa.stattools import adfuller
result = adfuller(df['Average_cost'])
```

#### Seasonal Decomposition

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


![image](https://github.com/user-attachments/assets/3c9a4e42-1cf3-4772-adfb-346c62f69a1a)

he Prophet forecast indicates a stable upward trend from 2015 through 2021, with several changepoints identified throughout the timeline. Actual values during this period remain relatively flat, suggesting limited variability or possibly imprecise input data. After 2021, the forecast becomes highly erratic, with significant oscillations and widened confidence intervals, pointing to increased uncertainty in future predictions. These fluctuations may reflect an overfitted model or issues in seasonality assumptions. The presence of repeated sharp peaks and troughs suggests that the model is struggling to generalize beyond the training period. Additionally, the mismatch between actual and predicted values implies potential data quality or preprocessing problems. Overall, the model's performance is reliable during the historical window but needs refinement to produce stable future projections.



## Tools

- Python: pandas, numpy, seaborn, matplotlib, statsmodels, prophet
- Visualization: plotly
- Models: ARIMA, Prophet
- Feature Engineering: Differencing, Lagging, Scaling

---

## How to Run

1. Clone the repo
2. Add `Month_Value_1.csv` to the root folder
3. Install required packages:

```bash
pip install pandas numpy seaborn matplotlib statsmodels prophet plotly
```

4. Run the script or notebook
