
# Retail Sales Dataset Project

This project is part of the Data Science Certificate program at the University of Toronto’s Data Sciences Institute.

## Dataset

We are using the [Retail Sales Dataset](https://www.kaggle.com/datasets/mohammadtalib786/retail-sales-dataset), which contains transactional-level sales data including features such as date, product category, quantity, unit price, and revenue.

The raw dataset is stored in the `data/raw/` directory.

## Team Members

- Iryna Verbova ([iverbova](https://github.com/iverbova))
- Tetiana Hakansson ([t125yf](https://github.com/t125yf))
- Rabia Imra Kirli Ozis ([rabiaimra](https://github.com/rabiaimra))
- Ting Man([manntintin](https://github.com/manntintin))
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

##  Data Exploration

We take an initial look at the Retail Sales Dataset and provide valuable insights that will guide future analysis.

Objectives:

Examine the dataset's structure
Create exploratory visualizations
Produce statistical summaries
Identify insights to address business-related questions

- Our main goal is to explore how factors like product price, product type, age, and gender influence buying habits and total sales.

**Dataset Overview**

We worked with a clean dataset of 1,000 retail transactions. The key features included product information, price, quantity, and demographic data like age and gender.

- `Transaction ID`: Unique identifier per transaction
- `Date`: Transaction date
- `Customer ID`: Anonymized customer identifier
- `Gender`: Customer gender
- `Age`: Customer age
- `Product Category`: Product type (Beauty, Clothing, Electronics)
- `Quantity`: Number of items purchased
- `Price per Unit`: Cost per item
- `Total Amount`: Transaction value = `Quantity × Price per Unit`
-  All columns are well-formatted. There are no missing or duplicate values, and the calculated field `Total Amount` is consistent with `Quantity * Price per Unit`.

 **Exploratory Data Analysis (EDA)**
- This section explores key patterns in the dataset through visualizations:
- Sales trends
- Purchase behavior by category, gender, and age
- Average order values and price distributions

**Exploratory Insights**:
Understanding these patterns can help businesses adjust pricing, better segment customers, and tailor campaigns more effectively.

 Sales Trend: Sales fluctuate across dates with noticeable spikes around mid-year:

![alt text](images/01_data_exploration/sales_trend_over_time.png)

Product Category: Electronics contributes the highest revenue, followed by Clothing and Beauty:
![alt text](images/01_data_exploration/total_sales_by_product_category.png)
![alt text](images/01_data_exploration/sales_distribution_by_product_category_pie.png)

[![Preview](images/01_data_exploration/Piechart_preview.png)](images/01_data_exploration/sales_distribution_by_product_category_pie.png)

Gender Split: Females account for slightly higher overall sales compared to males:
- ![alt text](images/01_data_exploration/total_sales_by_gender.png)
  ![alt text](images/01_data_exploration/sales_distribution_by_gender.png)

Age Groups: Customers aged 40-60 make the largest contribution, followed by those aged 25-40, who also significantly contribute to sales volume:
    ![alt text](images/01_data_exploration/total_sales_by_age_group.png)
     ![alt text](images/01_data_exploration/sales_pie_by_age_group.png)
  
Average Order Value (AOV):  $456.00.
![alt text](images/01_data_exploration/AOV.png)

Price per Unit: Beauty items have the highest average price per unit, while Clothing is the most affordable category.
 ![alt text](images/01_data_exploration/average_price_per_unit_by_product_category.png)

These findings will guide our feature engineering and modeling decisions in the next phase.

 
## Feature Engineering
 Turning Raw Data into Insightful Features
 
 Before a model can make good predictions, we need to give it the right data — this process is called feature engineering. We transformed basic retail data into useful features.
These new features help our models recognize patterns more clearly — like spotting who’s a likely big spender or when sales spike during the week.

-We will generate all relevant features needed for our classification and regression models, including:
- Categorical encodings
- Temporal features
- Interaction-ready variables
- Target variable for classification

At the end, we’ll export a clean `processed_data.csv` to be used for modeling.

We grouped ages, tracked purchase timing, and tagged customers as high or low spenders — all of which helps our model understand behavior.

To simplify linear regression modeling, we also create numeric encodings for:
- Gender: Male = 0, Female = 1
- Age Group: `<25` = 1, `25-40` = 2, `40-60` = 3, `60+` = 4
- Product Category: Clothing = 1, Electronics = 2, Beauty = 3

| Feature                         | Type                                     | Reason                                                                                 |
|---------------------------------|------------------------------------------|----------------------------------------------------------------------------------------|
| Age Group                       | Categorical                              | Needed to evaluate interaction effects between age and price in regression. Binned into <25, 25-40, 40-60, 60+ for business relevance. |
| High Spender                    | Binary                                   | Target variable for classification. Labeled as 1 if the transaction is in the top 25% of Total Amount. |
| Month                            | Numeric                                  | Temporal feature to explore monthly patterns or seasonality.                            |
| Day of Week                     | Numeric                                  | Helps identify weekday/weekend trends. Can be used to enrich predictions.               |
| Avg Price per Item              | Numeric                                  | Provides insight into pricing behavior per transaction.                                |
| Gender_*, ProductCategory_*, AgeGroup_* | One-hot encoded categorical vars         | Useful for classification models and non-linear ML algorithms. `drop_first=True` used to avoid dummy variable trap. |
| Gender_Num, AgeGroup_Num, ProductCategory_Num | Numeric (label encoded)               | Added to support regression models (linear models often benefit from single numeric representations of categories). |

 Example: 
From original:

Age = 43

Gender = Female

Product = Electronics

After Transformation:

Age Group = 40–60

Gender = 1

Product Code = 2

This is what feature engineering looks like — we convert human-friendly info into model-friendly numbers.

Our goal is to teach the model to recognize which customers are likely to spend more — so we created this target feature.

With our engineered dataset, we can move confidently into building models that provide useful insights for decision-making.


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


## Classification Model
To classify high-spending customers based on demographics (gender, age group) and purchase attributes (product category and quantity) using the K-Nearest Neighbors algorithm.

We are using the feature-engineered dataset that includes encoded categorical variables, numerical features, and a binary target variable High Spender.
High Spender Label: Top 25% of customers labeled as "High Spender"

Split Data: 75% for training the model, 25% for testing

Standardization: Adjusted features so they are comparable (important for distance-based models like KNN)
We defined a cutoff point (75th percentile):

Customers who spent more than this value were labeled as High Spenders (1)
Everyone else was labeled Not High Spenders (0)

This gave us a binary classification task. 

KNN compares distances between customers — it’s sensitive to feature scales.

So we standardized features using Z-score normalization: z= (x−μ)/σ 
 
This ensures no feature (like Quantity) dominates others (like Gender).
**Best value for K:**

To find the best number of neighbors (k), we plot the error rate for values between 1 and 20.
This helps us select the optimal k that balances bias and variance.

![alt text](images/error_rate_vs_k_value.jpg)

The best value of k is: 2

**Train KNN**

We create a KNN classifier using k neighbors after finding the best value for K by Hyperparameter Tuning.
KNN works by comparing each test point to the k closest training points and assigning the majority class among them.
**KNN Distance Calculations:**

Compute distance to all training points using Euclidean distance formula:

distance: $d = \sqrt{(x_1 - x_1')^2 + (x_2 - x_2')^2 + \dots + (x_n - x_n')^2}$

It picks K closest points from training set using the above distance.

For classification, KNN checks the most frequent class among the k nearest neighbors.

**Example:**

If k=5 and the nearest neighbors have labels [1, 0, 1, 1, 0] → the predicted label is 1 (majority).

In case of a tie, behavior depends on the implementation (e.g., some libraries break ties by choosing the class with the lower label).

**Confusion Matrix**

![alt text](images/Confusion_matrix.png)
- This heatmap shows how well the classifier predicted the two classes:

True Positives (TP): 11 - Correctly predicted high spenders

True Negatives (TN): 191 - Correctly predicted non-high spenders

False Positives (FP): 9 - Non-high spenders misclassified as high

False Negatives (FN): 39 - High spenders missed by the model

It helps identify if the model is biased toward one class or struggles with imbalanced data.

Model Performance Summary

 Metric                              | Value                                                      |
| ----------------------------------- | ---------------------------------------------------------- |
| Best `k`                            | **2**                                                      |
| Accuracy                            | **80.8%**                                                  |
| Precision (High Spenders - Class 1) | **55%**                                                    |
| Recall (High Spenders - Class 1)    | **22%**                                                    |
| F1 Score (High Spenders)            | **31%**                                                    |

**Main Takeaways**:

The model achieves a solid overall accuracy of 81%, showing it generally performs well across the board.

However, it struggles with identifying actual high spenders, with a recall of only 22% — meaning many true high spenders go undetected. The model tends to be cautious in labeling someone as a high spender.

When it does predict a customer is a high spender, it's correct about 55% of the time — this is the precision.

The model leans toward predicting the majority class (non-high spenders), which is understandable given the imbalance in the data (only 50 out of 250 test cases are high spenders).

Conclusion:

## Conclusion

This project provided a comprehensive exploration of retail transaction data to uncover purchasing patterns and build predictive models for both purchase quantity and customer spending behavior.

- Our exploratory data analysis revealed clear trends in customer behavior by product category, gender, and age group. Electronics generated the most revenue, females slightly outspent males, and customers aged 40–60 emerged as the most active buyers. These insights are valuable for segmentation and marketing strategy.

- Through feature engineering, we transformed raw retail data into meaningful variables, enabling more effective modeling. We encoded categorical variables, created temporal and demographic groupings, and defined a high-spending target variable, setting the foundation for both regression and classification tasks.

- The regression model, built to predict quantity purchased, highlighted some statistically significant effects (such as price and age interactions), but overall showed weak explanatory power (R² = 0.014). This suggests that quantity decisions are likely influenced by additional unobserved factors not captured in the current dataset.

- The classification model, aimed at identifying high spenders using the K-Nearest Neighbors algorithm, achieved 81% overall accuracy, but demonstrated limited effectiveness in identifying true high spenders, with a recall of only 22%. The model favored the majority class (non-high spenders), a common issue when dealing with imbalanced data.


In summary, this project shows that leveraging data-driven marketing and advanced machine learning can greatly enhance the efficiency and impact of financial marketing efforts, leading to improved business results and more personalized client interactions.
