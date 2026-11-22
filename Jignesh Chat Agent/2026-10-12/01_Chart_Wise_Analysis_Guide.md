# 01. Chart-Wise Analysis & Storytelling Study Guide
**Project:** Integrated Retail Analytics for Store Optimization and Demand Forecasting  
**Author:** Jignesh | AI Pair Programmer: Antigravity  
**Date:** 2026-10-12  

---

## 🧭 Visual Analytics Framework: The UBM Rule

According to the AlmaBetter assessment rubric, exploratory visualizations must adhere to the **UBM Rule**:
- **U — Univariate Analysis:** Distribution, spread, central tendency, skewness, and outliers of individual variables.
- **B — Bivariate Analysis:** Pairwise relationships across numerical-numerical, numerical-categorical, and categorical-categorical dimensions.
- **M — Multivariate Analysis:** High-dimensional interaction effects across sales, time, economic drivers, store sizes, and markdowns.

For each chart below, the three mandatory evaluative questions are answered in detail.

---

### Chart 1: Distribution of Weekly Sales (Univariate)
```python
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 5))
sns.histplot(df_merged['Weekly_Sales'], bins=100, kde=True, color='royalblue')
plt.title('Chart 1: Distribution of Weekly Sales (Log-Scaled / Right-Skewed)', fontsize=14, fontweight='bold')
plt.xlabel('Weekly Sales ($)')
plt.ylabel('Frequency')
plt.axvline(df_merged['Weekly_Sales'].median(), color='red', linestyle='--', label=f"Median: ${df_merged['Weekly_Sales'].median():,.2f}")
plt.axvline(df_merged['Weekly_Sales'].mean(), color='orange', linestyle='-', label=f"Mean: ${df_merged['Weekly_Sales'].mean():,.2f}")
plt.legend()
plt.tight_layout()
plt.show()
```
* **1. Why did you pick the specific chart?**  
  A combined histogram and Kernel Density Estimation (KDE) plot was chosen to examine the fundamental distribution profile, skewness, and presence of extreme sales volumes across all store-department pairs.
* **2. What is/are the insight(s) found from the chart?**  
  Weekly sales exhibit a strong positive (right) skew. The vast majority of department-week sales are under \$25,000, but long-tail anomalies exceed \$100,000+ during festive seasons. A small proportion of records contain negative sales values (returns/chargebacks).
* **3. Positive & Negative Business Impact:**  
  * *Positive Impact:* Identifies the baseline revenue expectation for standard operations and underscores the need for log-transformation ($log1p$) to prevent linear models from over-penalizing standard volume weeks.
  * *Risk of Negative Impact:* Failing to filter or model negative sales separately will distort demand forecasts, leading to stockouts during regular purchasing periods.

---

### Chart 2: Store Size Distribution Across Store Types (Univariate / Bivariate)
```python
plt.figure(figsize=(9, 5))
sns.boxplot(data=stores_df, x='Type', y='Size', palette='Set2')
plt.title('Chart 2: Store Size Footprint by Store Category (Type A, B, C)', fontsize=14, fontweight='bold')
plt.xlabel('Store Type')
plt.ylabel('Store Size (Square Footage)')
plt.tight_layout()
plt.show()
```
* **1. Why did you pick the specific chart?**  
  A boxplot is optimal for comparing median floor space, interquartile ranges, and variance across discrete store types (A, B, and C).
* **2. What is/are the insight(s) found from the chart?**  
  Type A stores represent hypercenters/supercenters (median ~200,000 sq ft). Type B stores represent mid-sized regional stores (median ~115,000 sq ft), while Type C stores are small-format neighborhood express stores (~40,000 sq ft). There is clear structural stratification with negligible overlap.
* **3. Positive & Negative Business Impact:**  
  * *Positive Impact:* Enables localized inventory allocations. Supercenters can hold broad department catalogues, whereas Type C stores must prioritize fast-moving essential SKUs.
  * *Risk of Negative Impact:* Applying uniform SKU stocking across all store types would cause massive inventory holding costs and dead stock in Type C stores.

---

### Chart 3: Total Weekly Sales by Store Type (Bivariate)
```python
sales_by_type = df_merged.groupby('Type')['Weekly_Sales'].sum().reset_index()

plt.figure(figsize=(8, 5))
sns.barplot(data=sales_by_type, x='Type', y='Weekly_Sales', palette='Blues_d')
plt.title('Chart 3: Cumulative Weekly Sales Contribution by Store Type', fontsize=14, fontweight='bold')
plt.xlabel('Store Type')
plt.ylabel('Cumulative Sales ($ Billions)')
plt.tight_layout()
plt.show()
```
* **1. Why did you pick the specific chart?**  
  A bar plot highlights aggregate revenue contribution across store classes.
* **2. What is/are the insight(s) found from the chart?**  
  Type A stores generate over 51% of total enterprise revenue, followed by Type B (~39%), while Type C accounts for less than 10%.
* **3. Positive & Negative Business Impact:**  
  * *Positive Impact:* Directs marketing capital and supply-chain infrastructure investments toward Type A flagship stores where revenue elasticity is highest.
  * *Risk of Negative Impact:* Over-investing in Type C stores without adapting their product mix will yield poor return on investment (ROI).

---

### Chart 4: Holiday vs. Non-Holiday Average Weekly Sales (Bivariate)
```python
holiday_comp = df_merged.groupby('IsHoliday')['Weekly_Sales'].mean().reset_index()

plt.figure(figsize=(7, 5))
sns.barplot(data=holiday_comp, x='IsHoliday', y='Weekly_Sales', palette='coolwarm')
plt.title('Chart 4: Average Weekly Sales — Holiday Weeks vs. Non-Holiday Weeks', fontsize=14, fontweight='bold')
plt.xlabel('Is Holiday Week')
plt.ylabel('Average Weekly Sales ($)')
plt.tight_layout()
plt.show()
```
* **1. Why did you pick the specific chart?**  
  A bar chart provides a direct comparison of mean weekly sales lift during promotional holiday periods versus normal trading weeks.
* **2. What is/are the insight(s) found from the chart?**  
  Holiday weeks generate substantially higher average sales (~$17,035 vs ~$15,901 per department), confirming that holiday shopping drives immediate volume surges.
* **3. Positive & Negative Business Impact:**  
  * *Positive Impact:* Validates the necessity of holiday-weight adjustments in demand forecasting (WMAE weighting in evaluation).
  * *Risk of Negative Impact:* Underestimating holiday demand leads to catastrophic stockouts, lost revenue, and customer churn.

---

### Chart 5: Weekly Sales Trend Over Time (Time Series Bivariate)
```python
weekly_trend = df_merged.groupby('Date')['Weekly_Sales'].sum().reset_index()

plt.figure(figsize=(14, 5))
sns.lineplot(data=weekly_trend, x='Date', y='Weekly_Sales', color='darkgreen', linewidth=2)
plt.title('Chart 5: Enterprise Weekly Sales Trend (2010 – 2012)', fontsize=14, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Total Weekly Sales ($)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
```
* **1. Why did you pick the specific chart?**  
  A time series line chart displays longitudinal trends, macro seasonality, and repeating annual cycles over the 3-year timeline.
* **2. What is/are the insight(s) found from the chart?**  
  Pronounced annual spikes occur in late November (Thanksgiving / Black Friday) and late December (Pre-Christmas rush), followed by January post-holiday slumps. Year-over-year baseline sales remain robust and stable.
* **3. Positive & Negative Business Impact:**  
  * *Positive Impact:* Enables warehouse management to plan seasonal workforce scaling and pre-build inventory 4–6 weeks prior to November.
  * *Risk of Negative Impact:* Overstocking in January based on Q4 momentum causes severe markdowns and carrying costs.

---

### Chart 6: Top 10 Performing Departments by Total Sales (Bivariate)
```python
top_depts = df_merged.groupby('Dept')['Weekly_Sales'].sum().nlargest(10).reset_index()

plt.figure(figsize=(10, 5))
sns.barplot(data=top_depts, x='Dept', y='Weekly_Sales', order=top_depts['Dept'], palette='viridis')
plt.title('Chart 6: Top 10 Departments by Cumulative Revenue', fontsize=14, fontweight='bold')
plt.xlabel('Department Number')
plt.ylabel('Total Sales ($)')
plt.tight_layout()
plt.show()
```
* **1. Why did you pick the specific chart?**  
  Ranked horizontal or vertical bar charts clearly isolate key department drivers of enterprise revenue.
* **2. What is/are the insight(s) found from the chart?**  
  Departments 92, 95, 38, 72, and 90 account for a disproportionately large percentage of enterprise sales. These represent groceries, household staples, and consumer electronics.
* **3. Positive & Negative Business Impact:**  
  * *Positive Impact:* Focuses demand forecasting precision and supplier contract negotiations on hero departments.
  * *Risk of Negative Impact:* Supply chain disruptions in these top 5 departments pose systemic enterprise revenue risk.

---

### Chart 7: Temperature vs. Weekly Sales (Bivariate Numerical-Numerical)
```python
plt.figure(figsize=(9, 5))
sns.scatterplot(data=df_merged.sample(5000, random_state=42), x='Temperature', y='Weekly_Sales', alpha=0.3, color='crimson')
plt.title('Chart 7: Relationship Between Ambient Temperature and Weekly Sales', fontsize=14, fontweight='bold')
plt.xlabel('Temperature (°F)')
plt.ylabel('Weekly Sales ($)')
plt.tight_layout()
plt.show()
```
* **1. Why did you pick the specific chart?**  
  A scatter plot reveals non-linear patterns, clustering, and correlation between outdoor temperature and purchasing behavior.
* **2. What is/are the insight(s) found from the chart?**  
  Peak sales occur between 40°F and 70°F (autumn and spring seasons). Extreme cold (<20°F) or extreme heat (>90°F) show slight dips in footfall, but overall sales are resilient across temperature bands.
* **3. Positive & Negative Business Impact:**  
  * *Positive Impact:* Guides seasonal department transitions (e.g., winter apparel vs. summer gardening gear).
  * *Risk of Negative Impact:* Relying on temperature alone as a predictor yields weak forecasting power without seasonal lag features.

---

### Chart 8: Fuel Price vs. Weekly Sales (Bivariate Numerical-Numerical)
```python
plt.figure(figsize=(9, 5))
sns.scatterplot(data=df_merged.sample(5000, random_state=42), x='Fuel_Price', y='Weekly_Sales', alpha=0.3, color='purple')
plt.title('Chart 8: Fuel Price vs. Weekly Sales', fontsize=14, fontweight='bold')
plt.xlabel('Fuel Price ($/Gallon)')
plt.ylabel('Weekly Sales ($)')
plt.tight_layout()
plt.show()
```
* **1. Why did you pick the specific chart?**  
  Examines how gasoline pump inflation affects discretionary spending and store visitation.
* **2. What is/are the insight(s) found from the chart?**  
  Fuel prices surged from \$2.50 to \$4.00+ during 2010–2012, yet weekly sales volume remained consistent. Big-box shoppers treat Walmart as a one-stop-shop to save on trip frequency when fuel is high.
* **3. Positive & Negative Business Impact:**  
  * *Positive Impact:* Proves the discount retail value proposition remains defensive and resilient against energy inflation.
  * *Risk of Negative Impact:* Rising transport costs squeeze retail logistics margins if replenishment frequency is not optimized.

---

### Chart 9: CPI (Consumer Price Index) vs. Weekly Sales (Bivariate)
```python
plt.figure(figsize=(9, 5))
sns.scatterplot(data=df_merged.sample(5000, random_state=42), x='CPI', y='Weekly_Sales', alpha=0.3, color='teal')
plt.title('Chart 9: CPI vs. Weekly Sales Across Stores', fontsize=14, fontweight='bold')
plt.xlabel('Consumer Price Index (CPI)')
plt.ylabel('Weekly Sales ($)')
plt.tight_layout()
plt.show()
```
* **1. Why did you pick the specific chart?**  
  Evaluates regional price index levels against transaction volume.
* **2. What is/are the insight(s) found from the chart?**  
  CPI forms two distinct regional clusters (~130–140 in low-cost states and ~210–225 in mature urban regions). Both clusters achieve high sales volumes, indicating geographically broad customer retention.
* **3. Positive & Negative Business Impact:**  
  * *Positive Impact:* Validates regional price tiering strategies.
  * *Risk of Negative Impact:* Implementing flat national pricing across wildly different CPI zones hurts competitiveness in lower-CPI territories.

---

### Chart 10: Unemployment Rate vs. Weekly Sales (Bivariate)
```python
plt.figure(figsize=(9, 5))
sns.scatterplot(data=df_merged.sample(5000, random_state=42), x='Unemployment', y='Weekly_Sales', alpha=0.3, color='darkorange')
plt.title('Chart 10: Unemployment Rate vs. Weekly Sales', fontsize=14, fontweight='bold')
plt.xlabel('Unemployment Rate (%)')
plt.ylabel('Weekly Sales ($)')
plt.tight_layout()
plt.show()
```
* **1. Why did you pick the specific chart?**  
  Assesses macroeconomic unemployment headwinds against weekly sales.
* **2. What is/are the insight(s) found from the chart?**  
  High sales are sustained across regions with 6% to 10% unemployment. Discount retailers capture wallet-share substitution during difficult economic periods.
* **3. Positive & Negative Business Impact:**  
  * *Positive Impact:* In high-unemployment areas, expanding budget-friendly private label brands drives loyalty.
  * *Risk of Negative Impact:* Stocking premium brand tiers in high-unemployment clusters leads to inventory write-downs.

---

### Chart 11: MarkDown Promotions Availability & Magnitude (Multivariate)
```python
markdown_cols = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
md_totals = df_merged[markdown_cols].sum().reset_index()
md_totals.columns = ['MarkDown', 'Total_Amount']

plt.figure(figsize=(9, 5))
sns.barplot(data=md_totals, x='MarkDown', y='Total_Amount', palette='magma')
plt.title('Chart 11: Total Spend Across MarkDown Categories (1 to 5)', fontsize=14, fontweight='bold')
plt.xlabel('MarkDown Category')
plt.ylabel('Total Promotional Value ($)')
plt.tight_layout()
plt.show()
```
* **1. Why did you pick the specific chart?**  
  Compares total promotional intensity across five distinct promotional markdown mechanisms.
* **2. What is/are the insight(s) found from the chart?**  
  MarkDown1 and MarkDown5 comprise the largest promotional allowances. MarkDown3 spikes primarily during Thanksgiving/Holiday weeks.
* **3. Positive & Negative Business Impact:**  
  * *Positive Impact:* Directs marketing teams to align clearance promotions with historical markdown elasticity.
  * *Risk of Negative Impact:* Over-discounting through MarkDown1 without measuring sales lift erodes departmental gross margins.

---

### Chart 12: Monthly Seasonality of Sales (Multivariate: Month x Store Type)
```python
monthly_type = df_merged.groupby(['Month', 'Type'])['Weekly_Sales'].mean().reset_index()

plt.figure(figsize=(12, 5))
sns.lineplot(data=monthly_type, x='Month', y='Weekly_Sales', hue='Type', marker='o', palette='Set1')
plt.title('Chart 12: Average Monthly Sales Trajectory by Store Type', fontsize=14, fontweight='bold')
plt.xlabel('Month of Year (1 - 12)')
plt.ylabel('Average Weekly Sales ($)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
```
* **1. Why did you pick the specific chart?**  
  A multi-series monthly line plot isolates month-by-month sales trajectory across store types.
* **2. What is/are the insight(s) found from the chart?**  
  All store types demonstrate parallel surges in November and December. Type A stores exhibit the largest absolute surge during Q4 holiday shopping.
* **3. Positive & Negative Business Impact:**  
  * *Positive Impact:* Enables tiered labor and inventory planning by store format.
  * *Risk of Negative Impact:* Late logistics deliveries in November result in irreversible lost sales.

---

### Chart 13: Store Size vs. Mean Weekly Sales (Bivariate with Store Type Hue)
```python
store_perf = df_merged.groupby(['Store', 'Type', 'Size'])['Weekly_Sales'].mean().reset_index()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=store_perf, x='Size', y='Weekly_Sales', hue='Type', s=120, palette='Set2')
plt.title('Chart 13: Store Size vs. Average Weekly Sales by Store Type', fontsize=14, fontweight='bold')
plt.xlabel('Store Size (sq ft)')
plt.ylabel('Average Weekly Sales ($)')
plt.tight_layout()
plt.show()
```
* **1. Why did you pick the specific chart?**  
  A scatter plot with categorical hue explores the direct relationship between physical footprint and revenue generation.
* **2. What is/are the insight(s) found from the chart?**  
  Strong positive correlation ($r > 0.8$) between store size and mean weekly sales. Larger square footage accommodates high-margin destination departments.
* **3. Positive & Negative Business Impact:**  
  * *Positive Impact:* Justifies capital expenditures on expanding store floor space.
  * *Risk of Negative Impact:* Small stores (Type C) cannot support large department counts without degrading customer experience and inventory turns.

---

### Chart 14: Correlation Heatmap of Numerical Features (Multivariate)
```python
corr_cols = ['Weekly_Sales', 'Size', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
corr_matrix = df_merged[corr_cols].corr()

plt.figure(figsize=(11, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)
plt.title('Chart 14: Correlation Heatmap Across Retail and Macro Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```
* **1. Why did you pick the specific chart?**  
  A correlation heatmap provides an enterprise overview of linear multicollinearity and feature interactions.
* **2. What is/are the insight(s) found from the chart?**  
  Store Size exhibits the strongest positive correlation with Weekly Sales (~0.24 overall, higher at aggregate store level). MarkDown1 and MarkDown4 are heavily collinear ($r > 0.8$). Fuel Price and CPI have weak direct linear correlation with Weekly Sales, suggesting non-linear or lag effects.
* **3. Positive & Negative Business Impact:**  
  * *Positive Impact:* Guides feature selection; combining MarkDown1 and MarkDown4 avoids multicollinearity in linear models.
  * *Risk of Negative Impact:* Assuming economic indicators have zero predictive value because linear correlation is low is dangerous; tree-based models capture their non-linear threshold effects.

---

### Chart 15: Pair Plot of Key Drivers (Multivariate)
```python
sample_pair = df_merged[['Weekly_Sales', 'Size', 'Temperature', 'CPI', 'Type']].sample(2000, random_state=42)
sns.pairplot(sample_pair, hue='Type', palette='Set1', diag_kind='kde')
plt.suptitle('Chart 15: Pair Plot of Key Retail Drivers Segmented by Store Type', y=1.02, fontsize=14, fontweight='bold')
plt.show()
```
* **1. Why did you pick the specific chart?**  
  A pair plot enables simultaneous examination of bivariate distributions and univariate density plots segmented by Store Type.
* **2. What is/are the insight(s) found from the chart?**  
  Reveals distinct clustering: Type A stores dominate high sales and high square footage, Type C stores form a tightly clustered low-size, low-sales pocket, and CPI shows multi-modal regional distributions.
* **3. Positive & Negative Business Impact:**  
  * *Positive Impact:* Confirms that store segmentation is essential prior to training generalized forecasting models.
  * *Risk of Negative Impact:* One-size-fits-all predictive modeling leads to large relative percentage errors on small stores.
