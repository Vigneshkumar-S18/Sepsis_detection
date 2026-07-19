# THAARU Sepsis AI — Clinical Feature Engineering Report
## Derived Features & Model Schema Overview
**Generated:** July 19, 2026

---

## 1. Executive Summary
This report summarizes Phase 4 clinical feature engineering. All features are calculated patient-wise (no crossover patient leakage) and scaled symmetrically using parameters fit on the Training split cohort.

* **Final Features Shape:** 97 columns (68 original columns + 29 newly engineered features)
* **Training Records:** 1,086,436 rows
* **Validation Records:** 231,472 rows
* **Testing Records:** 234,302 rows

---

## 2. Derived Feature Groups

| Feature Group | Features Added | Clinical Justification |
| :--- | :--- | :--- |
| **Cardiovascular Indices** | Shock Index, Pulse Pressure, MAP deviation | Key flags for cardiovascular hypoperfusion and shock |
| **Clinical Ratios** | HR/MAP, Resp/O2Sat, HR/Temp | Tracks vital signs couplings under septic stress |
| **6h Rolling Statistics** | 6h Mean, 6h Std/Variability (HR, Temp, Resp, MAP, O2Sat) | Identifies vital signs instability trends |
| **1h Lag Differences** | 1h Slopes/Differences (HR, Temp, Resp, O2Sat) | Captures rate of change and acute worsening trends |
| **Sparsity Indicators** | Hours since Lactate, WBC, pH, Creatinine, etc. | Tracks density of clinician test ordering patterns |
| **ICU Timeline Flags** | First 24 Hours, Diurnal proxy hour of stay | Controls for stay duration and diurnal stay cycles |

---

## 3. Scale Statistics of Derived Features (Train Set Fit)
Below are the training-fit scaler parameters used to standardize the newly engineered features:

| Feature Name | Mean | Standard Deviation (Scale) | Variance |
| :--- | :--- | :--- | :--- |
| Shock_Index | 0.706 | 0.199 | 0.0396 |
| Pulse_Pressure | 60.0088 | 19.573 | 383.1012 |
| MAP_deviation | -1.1233 | 7.2356 | 52.3541 |
| HR_MAP_Ratio | 1.0569 | 0.2851 | 0.0813 |
| Resp_O2Sat_Ratio | 0.1928 | 0.0516 | 0.0027 |
| HR_Temp_Ratio | 2.2858 | 0.4564 | 0.2083 |
| HR_roll_mean_6h | 84.3492 | 15.7513 | 248.1026 |
| Resp_roll_mean_6h | 18.6217 | 3.9957 | 15.9654 |
| Temp_roll_mean_6h | 36.8701 | 0.6087 | 0.3705 |
| MAP_roll_mean_6h | 82.4139 | 13.4285 | 180.325 |
| O2Sat_roll_mean_6h | 97.289 | 2.1905 | 4.7981 |
| HR_roll_std_6h | 5.3448 | 4.274 | 18.2669 |
| Resp_roll_std_6h | 2.3981 | 1.7479 | 3.0551 |
| Temp_roll_std_6h | 0.2087 | 0.2123 | 0.0451 |
| MAP_roll_std_6h | 7.1337 | 4.7162 | 22.2427 |
| O2Sat_roll_std_6h | 1.2594 | 1.2105 | 1.4652 |
| HR_diff_1h | 0.0087 | 7.6761 | 58.9223 |
| Temp_diff_1h | 0.0028 | 0.2838 | 0.0806 |
| Resp_diff_1h | 0.0361 | 3.7439 | 14.017 |
| O2Sat_diff_1h | -0.0348 | 2.2094 | 4.8813 |
| hours_since_last_Lactate | 706.7053 | 449.1703 | 201753.9354 |
| hours_since_last_Creatinine | 220.1617 | 406.3733 | 165139.2294 |
| hours_since_last_WBC | 232.1916 | 414.1849 | 171549.1542 |
| hours_since_last_Platelets | 230.0279 | 412.7606 | 170371.3112 |
| hours_since_last_BUN | 211.6854 | 400.4394 | 160351.7196 |
| hours_since_last_pH | 549.8429 | 492.2981 | 242357.3933 |
| hours_since_last_HCO3 | 567.8827 | 491.269 | 241345.2427 |
| Diurnal_Proxy_Hour | 10.7376 | 6.6322 | 43.9867 |


---

## 4. Conclusion
The feature engineering stage is stable and complete. Processed splits now contain a comprehensive set of 94 features (including demographics, scaled vitals/labs, missingness flags, and derived clinical indices), ready for Phase 5 sequence formatting.