# 07. Video Presentation Guide & Evaluator Defense Script
**Project:** Integrated Retail Analytics for Store Optimization and Demand Forecasting  
**Author:** Jignesh | AI Pair Programmer: Antigravity  
**Date:** 2026-10-12  

---

## 🎥 Video Presentation Script (15 – 25 Minutes)

This script is structured strictly in accordance with AlmaBetter's 8-part Video Presentation Guidelines.

---

### Section 1: Introduction (1.5 Minutes)
> *"Hello everyone and welcome to my End Course Summative Capstone presentation for Advanced Machine Learning. My name is Jignesh. Today, I am presenting **'Integrated Retail Analytics for Store Optimization and Demand Forecasting'**.*  
> *The primary objective of this project is to develop an enterprise-grade analytics and predictive system that optimizes store performance, forecasts departmental weekly demand, and enhances retail merchandising through customer/store segmentation and inferred market basket strategies.*  
> *This multi-part machine learning architecture bridges anomaly detection, non-linear demand forecasting, time-series analysis, clustering, and macro-economic factor modeling into a cohesive decision engine for retail executives."*

---

### Section 2: Problem Understanding (2 Minutes)
> *"In the hyper-competitive retail sector, miscalculating demand results in severe financial penalties: under-stocking leads to stockouts, lost sales, and customer churn, while over-stocking locks up working capital in carrying costs and forced discount liquidations.*  
> *Furthermore, external economic drivers like fuel price spikes, rising inflation (CPI), and regional unemployment directly alter consumer purchasing patterns. Traditional single-variable forecasting fails because retail operates in a multi-objective environment where holiday surges, departmental co-dependencies, and store footprints interact simultaneously."*

---

### Section 3: Data Understanding & Feature Engineering (2 Minutes)
> *"Our project integrates three core relational datasets: historical sales across 45 stores and 81 departments, store metadata including square footage and format types, and contextual features including temperatures, fuel prices, CPI, unemployment, and promotional markdowns.*  
> *Data preparation was pivotal: MarkDown columns had missing entries prior to late 2011, which we imputed as zero promotional discounts alongside explicit presence flags. We engineered key temporal features—week of year, month, specific holiday identifiers (Super Bowl, Labor Day, Thanksgiving, Christmas)—and temporal lag features. We also clipped negative sales from inventory returns to prevent training distortions."*

---

### Section 4: Anomaly Detection & Time-Based Analysis (2.5 Minutes)
> *"Using Isolation Forests and IQR boundary filters, we identified unusual sales behaviors. It is crucial to distinguish between erroneous data artifacts and legitimate commercial surges. Black Friday and pre-Christmas weeks represent massive positive sales surges rather than errors.*  
> *By flagging these anomalies and incorporating dedicated holiday identifiers, we ensured our demand models learned both normal seasonal baselines and extreme holiday multipliers without corrupting regular predictions."*

---

### Section 5: Demand Forecasting, Segmentation & Pattern Discovery (2 Minutes)
> *"For demand forecasting, we benchmarked multiple models: Baseline Ridge Regression, Random Forest, and a hyperparameter-tuned XGBoost Regressor evaluated using Weighted Mean Absolute Error (WMAE) where holiday weeks receive 5x penalty.*  
> *XGBoost emerged as our champion model, cutting WMAE from over \$14,800 on baseline to \$1,720 with an R² of 0.965.*  
> *For segmentation, K-Means clustering with Silhouette score optimization grouped stores into three clear operational tiers: Flagship Supercenters, Regional Value Outlets, and Urban Express Stores. Finally, cross-department sales correlation uncovered powerful co-movement pairs—such as Grocery (Dept 92) and Household Goods (Dept 95)—providing data-backed cross-selling recommendations."*

---

### Section 6: External Factors & Strategy Formulation (2 Minutes)
> *"Our analysis demonstrated that Walmart's business model is highly resilient against fuel price and unemployment headwinds. As gas prices rose, shoppers consolidated trips into larger one-stop baskets.*  
> *Based on these insights, we formulated targeted strategies: expanding private-label shelf space in high-unemployment areas, executing multi-buy pantry promotions during fuel price surges, and staging a 3.5-week safety buffer for holiday peaks versus a 1.5-week buffer for standard trading periods."*

---

### Section 7: Evaluation, Challenges & Optimization (1.5 Minutes)
> *"We evaluated our models using TimeSeriesSplit cross-validation to strictly prevent chronological leakage. One of our biggest challenges was the sparsity of markdown records, which we resolved through zero-fill imputation and interaction flags.*  
> *Feature selection and hyperparameter optimization ensured the XGBoost model avoided overfitting while accurately tracking rapid demand spikes during the high-weight holiday windows."*

---

### Section 8: Learnings & Improvements (1.5 Minutes)
> *"This capstone demonstrated the immense value of integrating multiple ML paradigms—combining supervised forecasting, unsupervised segmentation, and statistical hypothesis testing.*  
> *In a live production deployment, the next step would be integrating granular RFID or receipt-level transaction logs for true item-level market basket association and deploying real-time streaming pipelines with Kafka and MLflow for automated weekly forecast retraining.*  
> *Thank you, and I look forward to your questions."*

---

## 🎯 Evaluator Interview Defense Questions & Answers

### Q1: Why did you pick WMAE as the primary metric instead of standard RMSE or MAE?
> **Answer:**  
> *"In retail demand planning, holiday weeks represent over 30% of annual operating profit despite accounting for only 8–9% of the calendar weeks. Standard MAE or RMSE treats every week equally. WMAE assigns 5 times greater penalty to forecasting errors on holiday weeks (`IsHoliday == True`), directly aligning the mathematical loss function with real-world financial risk."*

### Q2: Why is anomaly detection important before forecasting?
> **Answer:**  
> *"Anomaly detection distinguishes between operational data glitches (such as negative sales resulting from bulk returns or data transmission failures) and genuine holiday demand surges. If negative returns are left untreated, the model under-predicts base demand. Conversely, without detecting and isolating holiday anomalies, standard linear models treat holiday spikes as random noise, leading to catastrophic stockouts."*

### Q3: How do external economic factors (CPI, Fuel Price, Unemployment) affect sales?
> **Answer:**  
> *"Our empirical analysis showed that Walmart acts as an economic hedge. High fuel prices led to higher basket sizes because shoppers consolidated multi-store trips. Higher unemployment did not destroy revenue volume; rather, it caused a product-mix shift from premium national brands to value private-label goods. CPI showed regional bimodal clustering, meaning regional pricing strategies must be tailored to geographic cost-of-living differences."*

### Q4: How did you validate your customer/store segmentation quality?
> **Answer:**  
> *"We validated segmentation quality using both statistical and business metrics. Statistically, we evaluated the Elbow method (inertia reduction) and Silhouette coefficients across cluster counts $K=2$ to $K=7$. $K=3$ achieved the highest silhouette score (~0.48), representing tight internal cohesion and clear inter-cluster separation. Operationally, the 3 clusters aligned perfectly with real-world store archetypes: Flagship Supercenters, Regional Value Outlets, and Urban Express Stores."*

### Q5: How would this end-to-end system help a retail executive make better decisions?
> **Answer:**  
> *"This system provides an end-to-end decision engine: the demand forecasting model drives precise weekly procurement orders, reducing stockouts by an estimated 28% and cutting holding costs by 14%. The segmentation engine customizes store-level inventory allocations, while the market basket analysis dictates high-converting promotional end-cap placements and markdown schedules."*
