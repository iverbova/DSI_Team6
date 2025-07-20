
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

**01 - Data Exploration**

In this section, we take an initial look at the Retail Sales Dataset and provide valuable insights that will guide future analysis.

Objectives:

Examine the dataset's structure
Create exploratory visualizations
Produce statistical summaries
Identify insights to address business-related questions

Our main goal is to explore how factors like product price, product type, age, and gender influence buying habits and total sales.

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
  All columns are well-formatted. There are no missing or duplicate values, and the calculated field `Total Amount` is consistent with `Quantity * Price per Unit`.

 **Exploratory Data Analysis (EDA)**
Before modeling, we ensured our data was accurate. We removed duplicates, fixed missing values, and validated key calculations like price x quantity.
This section explores key patterns in the dataset through visualizations:
- Sales trends
- Purchase behavior by category, gender, and age
- Average order values and price distributions

**Exploratory Insights**:
Understanding these patterns can help businesses adjust pricing, better segment customers, and tailor campaigns more effectively.

- Sales Trend: Sales fluctuate across dates with noticeable spikes around mid-year:

![alt text](images/01_data_exploration/sales_trend_over_time.png)

Product Category: Electronics contributes the highest revenue, followed by Clothing and Beauty:
![alt text](images/01_data_exploration/total_sales_by_product_category.png)
![alt text](images/01_data_exploration/sales_distribution_by_product_category_pie.png)

- Gender Split: Females account for slightly higher overall sales compared to males:
- ![alt text](images/01_data_exploration/total_sales_by_gender.png)
  ![alt text](images/01_data_exploration/sales_distribution_by_gender.png)

Age Groups: Customers aged 40-60 make the largest contribution, followed by those aged 25-40, who also significantly contribute to sales volume:
    ![alt text](images/01_data_exploration/total_sales_by_age_group.png)
     ![alt text](images/01_data_exploration/sales_pie_by_age_group.png)
  
- Average Order Value (AOV):  $456.00.
![alt text](images/01_data_exploration/AOV.png)

- Price per Unit: Beauty items have the highest average price per unit, while Clothing is the most affordable category.
 ![alt text](images/01_data_exploration/average_price_per_unit_by_product_category.png)

These findings will guide our feature engineering and modeling decisions in the next phase.
  - 
 
**02 - Feature Engineering**
 Turning Raw Data into Insightful Features
 
 Before a model can make good predictions, we need to give it the right data — this process is called feature engineering. We transformed basic retail data into useful features.
These new features help our models recognize patterns more clearly — like spotting who’s a likely big spender or when sales spike during the week.

We will generate all relevant features needed for our classification and regression models, including:
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


**03 Regression Model**

To better understand the factors that drive product purchase quantity in a retail context, we developed a linear regression model using transactional data. The goal was to examine how unit price, customer demographics (age group and gender), and their interactions influence the quantity of items purchased in a single transaction.
The team built a linear regression model to explore what influences the number of items (Quantity) purchased in each retail transaction.

We wanted to understand what influences how many items a customer buys in a retail transaction.

We looked at:

The price of a product,

The age group and gender of the customer, and

How these factors might combine to affect shopping behavior.

This helps answer questions like:

Do younger people buy more when prices drop?

Are men and women equally sensitive to price?

Does price affect older customers differently?

Model formula:
Quantity = β0 + β1*(Price per Unit) + β2*(Age Group) + β3*(Gender) + β4*(Price per Unit*Age Group) + β5*(Price per Unit*Gender) + ε
Purpose: Understand how price impacts behavior across different groups.
This model allowed them to:

Measure direct effects: e.g., "Do older people buy more?"

Capture interaction effects: e.g., "Does price affect older people differently?"
They also split the data into training and test sets to check generalizability.

This model doesn’t just check average effects. It looks deeper into how price sensitivity changes depending on the shopper's age or gender.

From an **industry perspective**, this type of model provides valuable insights into price sensitivity across different customer segments. For example:

- Personalized pricing and promotions
- Demand forecasting
- Segmentation analysis
- Strategic decision-making

Key Findings:

 Insights:
Price per unit: Slightly higher prices are linked to buying more, not less — which is unusual, but might happen with luxury or premium products.

Age groups:

Customers aged 25–60 tend to buy more items than those under 25.

Gender: No clear difference — men and women bought similar quantities.

- `Price per Unit`: Coefficient = +0.0017, p = 0.007, t = 2.72
 
    Higher prices slightly increases Quantity on average. The effect is small but statistically significant at 1% level. 

- `Gender` (Female vs. Male): Not significant (p = 0.559, t = 0.58)

    No clear evidence of gender-based differences in purchase quantity.

- `Age Groups`:
 - `25-40`: Coefficient = +0.3823, p = 0.042, t = 2.04  -> Statistically significant at 5% level
 - `40-60`: Coefficient = +0.3639, p = 0.039, t = 2.07  -> Statistically significant at 5% level
 - `60+`: Not significant (p = 0.435)

    Customers aged 25-60 buy slightly more than those under 25.


- Interaction Terms:
 - `Price * Age 25-40`: Negative, p = 0.023, t = -2.29  -> Statistically significant at 5% level
 - `Price * Age 40-60`: Negative, p = 0.005, t = -2.84  -> Statistically significant at 1% level
 - `Price * Age 60+`: Not significant
 
    Higher prices reduce quantity more for age groups 25-60.
 
 - `Price * Gender`: Not significant


