# Import packages
import pandas as pd
import joblib
import streamlit as st

# Streamlit Page Configuration
st.set_page_config(page_title="Employee Attrition Prediction", layout="wide")

# Load model files
@st.cache_resource
def load_everything():
    df = pd.read_csv("employee_attrition_cleaned.csv")
    scaler = joblib.load("scaler.pkl")
    model_columns = joblib.load("model_columns.pkl")
    log_reg_model = joblib.load("lr_model.pkl")
    random_forest_model = joblib.load("rf_model.pkl")
    performance_model = joblib.load("perf_model.pkl")
    perf_features = joblib.load("perf_features.pkl")
    return df, scaler, model_columns, log_reg_model, random_forest_model, performance_model, perf_features


df, scaler, model_columns, log_reg_model, random_forest_model, performance_model, perf_features = load_everything()

st.title("Employee Attrition Prediction")
model_choice = st.selectbox("Model", ["Logistic Regression", "Random Forest"])

# SECTION 1: Predicting Employee Attrition (Turnover Prediction):
st.header("Predict Employee Attrition")
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", 18, 60, 30)
    monthly_income = st.number_input("Monthly Income", 1000, 20000, 5000)
    over_time = st.selectbox("Over Time", ["Yes", "No"])

with col2:
    job_satisfaction = st.slider("Job Satisfaction", 1, 4, 3)
    job_involvement = st.slider("Job Involvement", 1, 4, 3)
    years_at_company = st.number_input("Number of Years in Company", 0, 40, 5)
    years_since_promotion = st.number_input("Years Since Last Promotion", 0, 40, 1)

with col3:
    department = st.selectbox("Department", df["Department"].unique())
    marital_status = st.selectbox("Marital Status", df["MaritalStatus"].unique())
    business_travel = st.selectbox("Business Travel", df["BusinessTravel"].unique())

if st.button("Predict"):
    # Start with a row where every column is 0
    columns_needed = df.drop(columns=["Attrition", "AttritionEnc"]).columns
    new_row = {}
    for col in columns_needed:
        new_row[col] = 0

    # Now fill in the values the user picked
    new_row["Age"] = age
    new_row["MonthlyIncome"] = monthly_income
    new_row["JobSatisfaction"] = job_satisfaction
    new_row["JobInvolvement"] = job_involvement
    new_row["YearsAtCompany"] = years_at_company
    new_row["OverTime"] = over_time
    new_row["Department"] = department
    new_row["MaritalStatus"] = marital_status
    new_row["BusinessTravel"] = business_travel

    # Feature Engineering new rows
    tenure_bucket = pd.cut([years_at_company], bins=[-1, 2, 5, 10, 50], labels=["0-2", "3-5", "6-10", "11+"])[0]
    new_row["TenureCategories"] = tenure_bucket
    new_row["EngagementScore"] = job_involvement + job_satisfaction
    new_row["PromotionRatio"] = years_since_promotion / (years_at_company + 1)

    # Conver new_row to DF
    input_df = pd.DataFrame([new_row])

    # One-Hot Encoding
    input_df = pd.get_dummies(input_df, drop_first=True)

    # Column reindex with model
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    # Feature Scaling
    input_scaled = scaler.transform(input_df)

    # Model Selection
    if model_choice == "Random Forest":
        chosen_model = random_forest_model
        threshold = 0.25
    else:
        chosen_model = log_reg_model
        threshold = 0.5

    probability_of_leaving = chosen_model.predict_proba(input_scaled)[0][1]
    will_leave = int(probability_of_leaving >= threshold)

    # Result
    if age >= 35:
        age_description = "older"
    else:
        age_description = "younger"

    if job_satisfaction <= 2:
        satisfaction_description = "low"
    else:
        satisfaction_description = "high"

    if over_time == "Yes":
        overtime_description = "works overtime frequently"
    else:
        overtime_description = "does not work overtime often"

    result_sentence = (
        f"An employee who is {age} years of age ({age_description}), works in the {department} department, "
        f"has {satisfaction_description} job satisfaction, and {overtime_description} "
        f"might be more likely to leave the company."
    )

    if will_leave == 1:
        st.write(f"{result_sentence} Predicted probability of leaving: {probability_of_leaving:.0%}.")
    else:
        st.write(f"This employee's profile does not match the typical leaver pattern. "
                 f"Predicted probability of leaving: {probability_of_leaving:.0%}.")

# SECTION 2: Predict Performance Rating
st.header("Predict Performance Rating")
p1, p2, p3 = st.columns(3)

with p1:
    education = st.slider("Education (1=Below College ... 5=Doctor)", 1, 5, 3)
    job_involvement_perf = st.slider("Job Involvement", 1, 4, 3, key="perf_ji")

with p2:
    job_level = st.slider("Job Level", 1, 5, 2)
    monthly_income_perf = st.number_input("Monthly Income", 1000, 20000, 5000, key="perf_mi")

with p3:
    years_at_company_perf = st.number_input("Years at Company", 0, 40, 5, key="perf_yac")
    years_in_role = st.number_input("Years in Current Role", 0, 40, 3)

if st.button("Predict Performance Rating"):
    perf_input = pd.DataFrame([{
        "Education": education,
        "JobInvolvement": job_involvement_perf,
        "JobLevel": job_level,
        "MonthlyIncome": monthly_income_perf,
        "YearsAtCompany": years_at_company_perf,
        "YearsInCurrentRole": years_in_role,
    }])

    perf_input = perf_input[perf_features]

    perf_prediction = performance_model.predict(perf_input)[0]
    perf_probability = performance_model.predict_proba(perf_input)[0][1]

    if job_involvement_perf >= 3:
        involvement_description = "higher"
    else:
        involvement_description = "lower"

    if job_level >= 3:
        level_description = "higher"
    else:
        level_description = "lower"

    if perf_prediction == 1:
        rating_description = "higher"
        predicted_rating = 4
    else:
        rating_description = "standard"
        predicted_rating = 3

    result_sentence = (
        f"Employees with {years_at_company_perf} years at the company, {involvement_description} job involvement, "
        f"and {level_description} job levels are likely to receive {rating_description} performance ratings."
    )

    st.write(f"{result_sentence} Predicted rating: {predicted_rating} "
             f"(probability of rating 4: {perf_probability:.0%}).")