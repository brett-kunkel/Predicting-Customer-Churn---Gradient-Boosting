# Predicting-Customer-Churn---Gradient-Boosting
This project uses customer data from an insurance company to predict churn using advanced feature engineering and gradient boosting models. The goal was to develop a model that could effectively identify customers likely to leave, allowing the business to take proactive retention measures.

🔍 **Project Overview**    
The raw data was provided across four separate datasets:

Customer personal info

Contract information

Phone service details

Internet service details


These were merged on a shared customer_id column to create a unified dataset for modeling.

🧠 **Feature Engineering**  
Several features were engineered from the merged dataset to enhance model performance:

p_i_or_b: A categorical feature indicating whether the customer had phone service, internet service, or both.

customer_duration: The number of days between the customer’s start date and the reference date (date of churn or last contact).

To reduce the risk of data leakage and ensure the model predicts before churn occurs, a random float value between 12 and 15 was subtracted from the raw duration. This subtraction also introduced minor noise, which helped mitigate potential interactions with contract type (e.g., monthly, yearly, two-year) that might otherwise reveal the churn date indirectly.

Date features: The customer’s start date was transformed into several features:

Cyclical variables: Using sine and cosine transformations of the day of the year to capture seasonality.

Categorical variables: Such as start month, day of week, and quarter.

These engineered date features were tested in isolation and in combination with customer_duration to assess their impact on performance and data leakage risk.

⚠️ **Preventing Data Leakage**  
It was clear at the outset that most of the engineered begin_date features would lead to data leakage when used alongside the customer_duration feature, since combining both would allow the model to easily infer the churn label. However, it seemed plausible that certain features, such as start_dayofweek, might avoid this problem because they do not directly indicate the customer's start date. During experimentation, though, it became evident that even the start_dayofweek features allowed the model to narrow the range of potential start dates with enough precision to result in data leakage when combined with customer_duration. As a result, these features were also excluded when customer_duration was used in modeling.

🧪 **Models Used**  
XGBoost

LightGBM

CatBoost

📊 **Evaluation Metrics**  
ROC AUC (primary metric)

Accuracy

Recall

🔎 **Model Interpretability**  
SHAP (SHapley Additive exPlanations) was used to interpret model predictions and understand feature importance.

💼 **Potential Business Applications**  
Though no business-specific recommendations are included in this version, the SHAP visualizations provide actionable insights that could inform retention strategies, such as:

Identifying at-risk customer segments

Targeting promotional offers

📁 Repo Structure
```
customer-churn-prediction/
│
├── data/                          # Not included in repo, add your own
├── notebooks/                     # Jupyter notebooks for EDA and modeling
├── sprint_XX_env/                 # Local virtual environment (ignored)
├── src/                           # Modular Python scripts
│   ├── catboost_gpu_tuning.py     # Hyperparameter tuning for CatBoost
│   ├── lightgbm_gpu_tuning.py     # Hyperparameter tuning for LightGBM
│   ├── xgb_gpu_tuning.py          # Hyperparameter tuning for XGBoost
│   └── xgb_testing.py             # Evaluation script for XGBoost
├── .gitignore
├── Pipfile / environment.yml      # Dependency management
└── README.md
