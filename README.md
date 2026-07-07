# Employee Attrition Prediction

## Problem
Employee turnover poses a significant challenge for organizations, resulting in increased costs, reduced productivity, and team disruptions. Understanding the factors driving attrition and predicting at-risk employees is critical for effective retention strategies. This project aims to analyze employee data, identify key drivers of attrition, and build predictive models to support proactive decision-making in workforce management.

## Predictions
1. **Attrition** — will this employee leave? (Logistic Regression + Random Forest)
2. **Performance Rating** — Will they get Good Rating? (Decision Tree)

## Process
- Dropped columns that don't help: EmployeeNumber, EmployeeCount, Over18, StandardHours — same value for everyone or just an ID, so no signal
- One-hot encoded the text columns (drop_first=True) to avoid the dummy variable trap
- Made 3 new columns myself (feature engineering):
  - TenureCategories — grouped YearsAtCompany into buckets
  - EngagementScore — JobInvolvement + JobSatisfaction combined
  - PromotionRatio — years since promotion vs years at company
- Scaled the data (StandardScaler) after the train/test split, so there will be no leakage in test data
- Tried two models for Attrition: Logistic Regression and Random Forest
- Tuned the Random Forest with GridSearchCV
- Performance Rating only has two values in the data (3 and 4), so used Decision tree model.

## EDA
- People who work overtime leave a lot more
- Sales Dept, Sales Rep role, and HR field all have higher attrition
- Single employees leave more than married or divorced ones
- Younger employees (29-39) and people with 2-8 years at the company leave the most
- People who haven't been promoted in a while (<2 years) or are new to their role (<5 years) leave more

## Files
- `employee_attrition.py` - Data Loading, Cleaning, Preprocessing, EDA, Feature Engineering, Model Building, Evaluation, Pickle
- `streamlit_report.py` — Streamlit app runs from Pickel file (joblib)
- `Employee_Attrition.csv` — raw dataset
- `employee_attrition_cleaned.csv` — cleaned dataset
- `rf_model.pkl`, `lr_model.pkl`, `scaler.pkl`, `model_columns.pkl` — attrition model files
- `perf_model.pkl`, `perf_features.pkl` — performance rating model files

## How to Run
1. `pip install -r requirements.txt`
2. `python employee_attrition.py`
3. `streamlit run streamlit_report.py`

## Deployment
Streamlit link: https://employeeattrition-data.streamlit.app/
