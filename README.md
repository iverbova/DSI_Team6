
# Retail Sales Dataset Project

This project is part of the Data Science Certificate program at the University of Toronto’s Data Sciences Institute.

## Dataset

We are using the [Retail Sales Dataset](https://www.kaggle.com/datasets/mohammadtalib786/retail-sales-dataset), which contains transactional-level sales data including features such as date, product category, quantity, unit price, and revenue.

The raw dataset is stored in the `data/raw/` directory.

## Team Members

- Iryna Verbova ([iverbova](https://github.com/iverbova))
- Tetiana Hakansson ([t125yf](https://github.com/t125yf))
- Rabia Imra Kirli Ozis ([rabiaimra](https://github.com/rabiaimra))
- Ting Man
- Vrushali Patil ([vrushalipatil9763](https://github.com/vrushalipatil9763))

## Business Problem

This project investigates purchasing behavior in the context of a small to medium-sized retail business. We aim to explore how **customer demographics** (age, gender) and **product features** (category, price) affect purchasing patterns.

Specifically, we are exploring:
- How sensitive different age groups and genders are to **price changes**.
- Whether it's possible to **identify high-spending customers** based on transaction attributes.

The outcomes will help retail stakeholders make **more informed pricing decisions**, **segment customers**, and **design targeted marketing strategies**.

## Stakeholders

Our key stakeholders include:
- **Retail business owners and managers** who want to understand their customers better
- **Marketing teams** designing promotions and loyalty programs
- **Pricing analysts** seeking insight into price elasticity by segment

These stakeholders care about improving profitability, customer retention, and targeted customer experiences.

---

## Business Questions

1. **Regression Question**:  
   _Can we predict whether price per unit affects the quantity purchased differently across age groups or genders?_  
   → This helps us identify **price sensitivity** in various customer segments.

2. **Classification Question**:  
   _Can we classify whether a customer will be a high spender based on their demographics and purchase details?_  
   → This supports **targeted engagement** of high-value customers.

---

## Initial Dataset Review

- The dataset is **clean and well-structured**.
- No missing values or duplicates.
- `Total Amount` is consistent with `Quantity × Price per Unit`.
- Categorical fields (`Gender`, `Product Category`) are tidy and usable.
- Contains sufficient diversity in ages, product categories, and purchase amounts to pursue both regression and classification tasks.

## Methods & Technologies

- Python (pandas, NumPy)
- Data Visualization (matplotlib, seaborn)
- Feature Engineering (pandas, one-hot encoding, quantile thresholds)
- GitHub (collaboration and version control)
- Multiple Linear Regression Model (statsmodels, sklearn)
- 
- 

## Risks and Uncertainties

- Dataset may lack granularity for some modeling tasks
- Sample size (1,000 rows) may limit classification performance
- Potential for overfitting if too many engineered features are added
- The relationship between variables like age and quantity may be weak or non-linear

---

## Tasks & Timeline

| Team Member | Task |
|-------------|------|
| Iryna       | Data exploration, feature engineering |
| Imra        | Conducting regression modeling |
| Vrushali    | Conducting classification modeling |
| Tetiana     | Writing conclusions, maintaining narrative |
| Mandy       | Presentation development |

---

## Regression Model

To better understand the factors that drive product purchase quantity in a retail context, we developed a linear regression model using transactional data. The goal was to examine how unit price, customer demographics (age group and gender), and their interactions influence the quantity of items purchased in a single transaction.

**Our model:** 

Quantity = β0 + β1*(Price per Unit) + β2*(Age Group) + β3*(Gender) + β4*(Price per Unit*Age Group) + β5*(Price per Unit*Gender) + ε

This specification allows us to capture both the main effects of price, age, and gender, and how these effects interact — particularly, whether price sensitivity differs by demographic group.

The interaction terms capture differential price sensitivity, such as how price impacts might vary between age groups or between male and female customers. This allows the model to go beyond average effects and identify segment-specific behavioral patterns, which are often crucial for effective targeting and personalization in modern retail strategies.

From an **industry perspective**, this type of model provides valuable insights into price sensitivity across different customer segments. For example:

- Personalized pricing and promotions: Marketing teams can use insights from interaction terms to identify which segments (e.g., age 25–40) are more price-sensitive and target them with dynamic pricing or customized discounts.

- Demand forecasting: Understanding which demographic groups are more responsive to price helps inventory planners and analysts model future sales volumes more accurately under different pricing strategies.

- Segmentation analysis: The model provides evidence-based support for customer segmentation strategies by quantifying behavioral differences in purchase patterns across age and gender groups.

- Strategic decision-making: Retailers can use such models to inform store-level pricing, online targeting, and cross-channel optimization efforts based on predicted consumer responsiveness.

While the model’s predictive power is limited, it offers a foundational framework for interpreting customer behavior and can be expanded by incorporating additional variables such as seasonality, product-level features, loyalty data, or promotional history. In real-world applications, such models could be integrated into business intelligence dashboards or dynamic pricing systems to support evidence-based decision-making.


**OLS Regression Results Summary**

                            OLS Regression Results                            
==============================================================================
Dep. Variable:               Quantity   R-squared:                       0.014
Model:                            OLS   Adj. R-squared:                  0.002
Method:                 Least Squares   F-statistic:                     1.172
Date:                Fri, 18 Jul 2025   Prob (F-statistic):              0.310
Time:                        13:57:10   Log-Likelihood:                -1148.1
No. Observations:                 750   AIC:                             2316.
Df Residuals:                     740   BIC:                             2362.
Df Model:                           9                                         
Covariance Type:            nonrobust                                         
============================================================================================
                               coef    std err          t      P>|t|      [0.025      0.975]
--------------------------------------------------------------------------------------------
const                        2.1543      0.163     13.184      0.000       1.834       2.475
Price per Unit               0.0017      0.001      2.715      0.007       0.000       0.003
Gender_Num                   0.0658      0.113      0.584      0.559      -0.155       0.287
Age Group_25-40              0.3823      0.188      2.038      0.042       0.014       0.750
Age Group_40-60              0.3639      0.176      2.069      0.039       0.019       0.709
Age Group_60+                0.1678      0.215      0.781      0.435      -0.254       0.590
Price_Age_25_40             -0.0016      0.001     -2.285      0.023      -0.003      -0.000
Price_Age_40_60             -0.0019      0.001     -2.844      0.005      -0.003      -0.001
Price_Age_60+               -0.0010      0.001     -1.146      0.252      -0.003       0.001
Price_Gender_Interaction    -0.0003      0.000     -0.617      0.537      -0.001       0.001
==============================================================================
Omnibus:                     6041.210   Durbin-Watson:                   1.949
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               57.838
Skew:                           0.023   Prob(JB):                     2.76e-13
Kurtosis:                       1.640   Cond. No.                     2.63e+03
==============================================================================

**Model Objective:**

Predict Quantity purchased based on unit price, age group, gender, and their interactions.

**Model Fit:**
- R-squared: 0.014 → Model explains only 1.4% of variation in Quantity which shows weak explanatory power.

- Adjusted R-squared: 0.002

- F-statistic: 1.172, p-value: 0.310 → Model is not statistically significant overall. In other words, our predictors jointly do not explain the variation in Quantity at conventional significance levels (p > 0.05).

**Key Findings:**

- `Price per Unit`: Higher prices slightly increases quantity on average. The effect is small but statistically significant at 1% level. 

- `Gender` (Female vs. Male): No clear evidence of gender-based differences in purchase quantity.

- `Age Groups`: Customers aged 25–60 buy slightly more than those under 25.

- **Interaction Terms:**
- Higher prices reduce quantity more for age groups 25–60.

- No evidence of gender-based differences in price sensitivity.

- **Conclusions:**

 - Some individual coefficients, such as price per unit, age groups 25-40 and 40-60, and their interaction terms, are statistically significant which implies that they have significant explanatory power in explaining the quantity purchased.  

 - However, the overall model lacks predictive power with low R2 and F-statistic. Thus, our model is not explaining the variability in quantity well. Most of the variation is likely driven by factors not included in the model.

 - Interaction terms reveal age-based differences in price sensitivity.

 - Gender appears to have no meaningful effect on quantity purchased.

 - Interestingly, the regression model reports a positive and statistically significant main effect of price per unit, suggesting that - all else equal - higher-priced products are associated with larger quantities purchased. While this may seem counterintuitive from a classical demand perspective, it could reflect nonelastic or luxury-oriented purchasing behavior, or the presence of premium product categories where higher prices do not deter demand and may even be associated with perceived value.

 - However, the model also includes interaction terms between price and age group, and these show significant negative coefficients for customers aged 25–60. This indicates that for these segments, higher prices are associated with lower quantities purchased, aligning more closely with traditional downward-sloping demand behavior. In other words, price sensitivity does exist in the data, but it varies across customer segments.

- **Additional Notes**
- To assess the model's ability to generalize beyond the data it was trained on, we randomly split the dataset into a training set (75%) and a test set (25%). The results indicate that the model performs poorly on unseen data with considerable percentage error in predictions.

---