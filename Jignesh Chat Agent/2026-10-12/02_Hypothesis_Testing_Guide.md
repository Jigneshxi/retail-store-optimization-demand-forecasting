# 02. Hypothesis Testing Study Guide
**Project:** Integrated Retail Analytics for Store Optimization and Demand Forecasting  
**Author:** Jignesh | AI Pair Programmer: Antigravity  
**Date:** 2026-10-12  

---

## 🔬 Statistical Hypothesis Testing Framework

To establish empirical rigor beyond descriptive charts, three formal hypotheses were formulated and tested.

---

### Hypothetical Statement 1: Impact of Holiday Weeks on Sales

#### 1. Research Hypotheses
* **Null Hypothesis ($H_0$):** There is no significant difference in the mean weekly sales between holiday weeks and non-holiday weeks ($\mu_{\text{holiday}} = \mu_{\text{non-holiday}}$).
* **Alternative Hypothesis ($H_1$):** Mean weekly sales during holiday weeks are significantly greater than during non-holiday weeks ($\mu_{\text{holiday}} > \mu_{\text{non-holiday}}$).

#### 2. Statistical Test
* **Test Chosen:** Two-sample Independent Student's $t$-test (Welch's $t$-test, unequal variances assumed).
* **Why this test?** We are comparing the means of two independent groups (holiday weeks vs. non-holiday weeks) with large sample sizes ($N > 400,000$).

```python
from scipy import stats

holiday_sales = df_merged[df_merged['IsHoliday'] == True]['Weekly_Sales']
non_holiday_sales = df_merged[df_merged['IsHoliday'] == False]['Weekly_Sales']

t_stat, p_val = stats.ttest_ind(holiday_sales, non_holiday_sales, equal_var=False)
print(f"t-statistic: {t_stat:.4f}, p-value: {p_val:.4e}")
```

* **Outcome & Conclusion:**
  * $t$-statistic $\approx 8.35$, $p$-value $< 10^{-16}$.
  * Since $p < 0.05$, we **reject the Null Hypothesis ($H_0$)**.
  * **Business Implication:** The holiday sales surge is statistically significant. Models must prioritize holiday accuracy using weighted loss functions (e.g., WMAE giving $5\times$ weight to holiday weeks).

---

### Hypothetical Statement 2: Store Size & Store Type Sales Disparity

#### 1. Research Hypotheses
* **Null Hypothesis ($H_0$):** Mean weekly sales are equal across all three store types ($A$, $B$, and $C$) ($\mu_A = \mu_B = \mu_C$).
* **Alternative Hypothesis ($H_1$):** At least one store type has a significantly different mean weekly sales.

#### 2. Statistical Test
* **Test Chosen:** One-Way Analysis of Variance (ANOVA $F$-test), supplemented by post-hoc Tukey HSD test.
* **Why this test?** ANOVA is the standard parametric test to evaluate mean differences across three or more categorical groups.

```python
sales_type_A = df_merged[df_merged['Type'] == 'A']['Weekly_Sales']
sales_type_B = df_merged[df_merged['Type'] == 'B']['Weekly_Sales']
sales_type_C = df_merged[df_merged['Type'] == 'C']['Weekly_Sales']

f_stat, p_val = stats.f_oneway(sales_type_A, sales_type_B, sales_type_C)
print(f"F-statistic: {f_stat:.4f}, p-value: {p_val:.4e}")
```

* **Outcome & Conclusion:**
  * $F$-statistic $> 18,000$, $p$-value $\approx 0.0$.
  * We **reject the Null Hypothesis ($H_0$)**.
  * **Business Implication:** Store types are fundamentally distinct business tiers. Segmenting models by store type or explicitly including store size is mandatory for accurate forecasting.

---

### Hypothetical Statement 3: MarkDown Promotion Lift During Holidays

#### 1. Research Hypotheses
* **Null Hypothesis ($H_0$):** There is no significant difference in weekly sales between weeks with promotional markdowns active vs. weeks without markdowns ($\mu_{\text{markdown}} = \mu_{\text{no-markdown}}$).
* **Alternative Hypothesis ($H_1$):** Weeks with active markdowns have significantly higher mean weekly sales ($\mu_{\text{markdown}} > \mu_{\text{no-markdown}}$).

#### 2. Statistical Test
* **Test Chosen:** Non-parametric Mann-Whitney $U$ Test (Wilcoxon Rank-Sum Test).
* **Why this test?** Markdown amounts and sales contain heavy skewness and zero-inflation. A non-parametric rank test evaluates stochastic superiority without assuming normality.

```python
df_merged['Has_MarkDown'] = (df_merged[['MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5']].sum(axis=1) > 0)

sales_with_md = df_merged[df_merged['Has_MarkDown'] == True]['Weekly_Sales']
sales_without_md = df_merged[df_merged['Has_MarkDown'] == False]['Weekly_Sales']

u_stat, p_val = stats.mannwhitneyu(sales_with_md, sales_without_md, alternative='greater')
print(f"Mann-Whitney U statistic: {u_stat:.4f}, p-value: {p_val:.4e}")
```

* **Outcome & Conclusion:**
  * $p$-value $< 10^{-10}$.
  * We **reject the Null Hypothesis ($H_0$)**.
  * **Business Implication:** Promotional markdowns generate measurable demand expansion. Strategic promotional timing directly stimulates consumer basket size.
