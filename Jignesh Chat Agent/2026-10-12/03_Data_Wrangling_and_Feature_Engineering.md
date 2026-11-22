# 03. Data Wrangling & Feature Engineering Study Guide
**Project:** Integrated Retail Analytics for Store Optimization and Demand Forecasting  
**Author:** Jignesh | AI Pair Programmer: Antigravity  
**Date:** 2026-10-12  

---

## 🛠️ Data Wrangling Pipeline

The raw retail data is distributed across three separate relational files:
1. `sales data-set.csv`: 421,570 records (`Store`, `Dept`, `Date`, `Weekly_Sales`, `IsHoliday`)
2. `Features data set.csv`: 8,190 records (`Store`, `Date`, `Temperature`, `Fuel_Price`, `MarkDown1-5`, `CPI`, `Unemployment`, `IsHoliday`)
3. `stores data-set.csv`: 45 records (`Store`, `Type`, `Size`)

### 1. Merging & Schema Alignment
```python
import pandas as pd
import numpy as np

# Load datasets
sales = pd.read_csv('sales data-set.csv')
features = pd.read_csv('Features data set.csv')
stores = pd.read_csv('stores data-set.csv')

# Standardize Date format
sales['Date'] = pd.to_datetime(sales['Date'], format='%d/%m/%Y')
features['Date'] = pd.to_datetime(features['Date'], format='%d/%m/%Y')

# Merge into unified analytical dataframe
df_merged = sales.merge(features, on=['Store', 'Date', 'IsHoliday'], how='left')
df_merged = df_merged.merge(stores, on='Store', how='left')
df_merged.sort_values(by=['Store', 'Dept', 'Date'], inplace=True)
```

---

## 2. Handling Missing Values (MarkDown Imputation)

* **Challenge:** MarkDown fields (`MarkDown1` through `MarkDown5`) are heavily missing (~60% nulls) prior to November 2011 because retailers only began recording markdown campaign tracking data midway through the time series.
* **Solution:**
  1. Fill missing MarkDown values with `0.0` (indicating no promotional discount active).
  2. Create an explicit indicator column `Has_MarkDown` and `Total_MarkDown = MarkDown1 + ... + MarkDown5`.
  3. Replace any negative sales with 0 or flag them as customer return events.

```python
# MarkDown Imputation
md_cols = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
for col in md_cols:
    df_merged[col] = df_merged[col].fillna(0.0)

df_merged['Total_MarkDown'] = df_merged[md_cols].sum(axis=1)
df_merged['Has_MarkDown'] = (df_merged['Total_MarkDown'] > 0).astype(int)

# Interpolate any sparse missing economic indicators
df_merged['CPI'] = df_merged.groupby('Store')['CPI'].transform(lambda x: x.interpolate(method='linear').bfill().ffill())
df_merged['Unemployment'] = df_merged.groupby('Store')['Unemployment'].transform(lambda x: x.interpolate(method='linear').bfill().ffill())
```

---

## 3. Anomaly Detection & Outlier Handling

### Techniques Implemented:
1. **IQR Filtering:** Calculating Interquartile Range ($Q1 - 1.5 \times IQR$ and $Q3 + 1.5 \times IQR$) by department.
2. **Isolation Forest:** Multi-dimensional anomaly scoring taking into account `Weekly_Sales`, `Total_MarkDown`, `IsHoliday`, and `Store_Size`.

```python
from sklearn.ensemble import IsolationForest

iso_forest = IsolationForest(contamination=0.015, random_state=42)
features_for_anomaly = df_merged[['Weekly_Sales', 'Size', 'Total_MarkDown', 'Temperature']].copy()
df_merged['Is_Anomaly'] = iso_forest.fit_predict(features_for_anomaly)
# Anomaly flag: -1 is anomaly, 1 is normal
df_merged['Is_Anomaly'] = np.where(df_merged['Is_Anomaly'] == -1, 1, 0)
```

* **Business Insight:** True anomalies correspond directly to Black Friday promotional peaks ($> \$150,000$) or inventory returns (negative sales). We preserve genuine holiday spikes to train the demand forecast while clipping extreme return artifacts.

---

## 4. Advanced Feature Engineering

### Temporal Features:
```python
df_merged['Year'] = df_merged['Date'].dt.year
df_merged['Month'] = df_merged['Date'].dt.month
df_merged['Week'] = df_merged['Date'].dt.isocalendar().week
df_merged['Day'] = df_merged['Date'].dt.day
df_merged['Is_Month_End'] = df_merged['Date'].dt.is_month_end.astype(int)
```

### Specific Holiday Categorization:
Rather than a single binary flag, we classify individual cultural shopping holidays:
* **Super Bowl:** Mid-February (Week 6)
* **Labor Day:** Early September (Week 36)
* **Thanksgiving / Black Friday:** Late November (Week 47)
* **Christmas:** Late December (Week 51/52)

```python
def classify_holiday(row):
    if not row['IsHoliday']:
        return 'None'
    w = row['Week']
    if w in [6, 7]:
        return 'Super_Bowl'
    elif w in [36, 37]:
        return 'Labor_Day'
    elif w in [47, 48]:
        return 'Thanksgiving'
    elif w in [51, 52]:
        return 'Christmas'
    return 'Other_Holiday'

df_merged['Holiday_Name'] = df_merged.apply(classify_holiday, axis=1)
```

### Lagged Sales & Rolling Statistics:
```python
# Shifted 1-week, 2-week, 4-week, 52-week (Year-over-Year) lags per Store-Dept
df_merged['Sales_Lag_1'] = df_merged.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(1)
df_merged['Sales_Lag_4'] = df_merged.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(4)
df_merged['Sales_Lag_52'] = df_merged.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(52)

# Rolling 4-week mean and standard deviation
df_merged['Rolling_Mean_4'] = df_merged.groupby(['Store', 'Dept'])['Weekly_Sales'].transform(lambda x: x.shift(1).rolling(4).mean())
df_merged['Rolling_Std_4'] = df_merged.groupby(['Store', 'Dept'])['Weekly_Sales'].transform(lambda x: x.shift(1).rolling(4).std())
```

---

## 5. Categorical Encoding & Train/Test Splitting

* **Categorical Encoding:** One-Hot Encoding for `Type` and `Holiday_Name`.
* **Data Splitting Strategy:** **Time-Series Split** (chronological order, train on 2010–2011, test on 2012). This prevents data leakage and mimics real-world production forecasting.
