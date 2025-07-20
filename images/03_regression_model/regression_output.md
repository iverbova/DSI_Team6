**OLS Regression Results**

| **Metric**         | **Value**     | **Metric**          | **Value**   |
|--------------------|---------------|---------------------|-------------|
| Dep. Variable      | Quantity      | R-squared           | 0.014       |
| Model              | OLS           | **Adj. R-squared**  | **0.002**   |
| Method             | Least Squares | F-statistic         | 1.172       |
| Date               | Fri, 18 Jul 2025 | Prob (F-statistic) | 0.310       |
| Time               | 19:39:29      | Log-Likelihood      | -1148.1     |
| No. Observations   | 750           | AIC                 | 2316        |
| Df Residuals       | 740           | BIC                 | 2362        |
| Df Model           | 9             | Covariance Type     | nonrobust   |
                                        
---

**Coefficients Table**

| Variable                 | coef   | std err | t      | P>\|t\| | [0.025   | 0.975]   |
|--------------------------|--------|---------|--------|---------|-----------|----------|
| const                    | 2.1543 | 0.163   | 13.184 | 0.000   | 1.834     | 2.475    |
| Price per Unit           | 0.0017 | 0.001   | 2.715  | 0.007   | 0.000     | 0.003    |
| Gender_Num               | 0.0658 | 0.113   | 0.584  | 0.559   | -0.155    | 0.287    |
| Age Group_25-40          | 0.3823 | 0.188   | 2.038  | 0.042   | 0.014     | 0.750    |
| Age Group_40-60          | 0.3639 | 0.176   | 2.069  | 0.039   | 0.019     | 0.709    |
| Age Group_60+            | 0.1678 | 0.215   | 0.781  | 0.435   | -0.254    | 0.590    |
| Price_Age_25_40          | -0.0016| 0.001   | -2.285 | 0.023   | -0.003    | -0.000   |
| Price_Age_40_60          | -0.0019| 0.001   | -2.844 | 0.005   | -0.003    | -0.001   |
| Price_Age_60+            | -0.0010| 0.001   | -1.146 | 0.252   | -0.003    | 0.001    |
| Price_Gender_Interaction | -0.0003| 0.000   | -0.617 | 0.537   | -0.001    | 0.001    |

---

**Model Diagnostics**

| **Test**            | **Value**  | **Test**          | **Value**  |
|---------------------|------------|-------------------|------------|
| Omnibus             | 6041.210   | Durbin-Watson     | 1.949      |
| Prob(Omnibus)       | 0.000      | Jarque-Bera (JB)  | 57.838     |
| Skew                | 0.023      | Prob(JB)          | 2.76e-13   |
| Kurtosis            | 1.640      | Cond. No.         | 2.63e+03   |


Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 2.63e+03. This might indicate that there are
strong multicollinearity or other numerical problems.
