
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
- 
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