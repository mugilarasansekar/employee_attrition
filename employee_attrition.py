# Import packages
import pandas as pd
import joblib
import json
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# STEP 1: Load the data and drop columns that don't need
df = pd.read_csv("Employee_Attrition.csv")

# Dropping columns which have same values
df.drop(columns=["EmployeeNumber", "Over18", "StandardHours", "EmployeeCount"], inplace=True)

# Encoding Attrition
df["AttritionEnc"] = df["Attrition"].map({
    "Yes": 1,
    "No": 0
})

# STEP 2: Feature Engineering

# TenureCategories
df["TenureCategories"] = pd.cut(df["YearsAtCompany"], bins=[-1, 2, 5, 10, 50], labels=["0-2", "3-5", "6-10", "11+"])

# Engagement Score using JobInvolvement and JobSatisfaction
df["EngagementScore"] = df["JobInvolvement"] + df["JobSatisfaction"]

# Promotion Ratio with YearsAtCompany
# (Adding 1 for YearsAtCompany, since there are 0 years)
df["PromotionRatio"] = df["YearsSinceLastPromotion"] / (df["YearsAtCompany"] + 1)

# STEP 3: Model building

# Feature and Target
X = df.drop(columns=["Attrition", "AttritionEnc"])
y = df["AttritionEnc"]

# One-Hot encoding
X = pd.get_dummies(X, drop_first=True)

# Column names after One-Hot encoding
model_columns = X.columns.tolist()

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# STEP 4: Logistic Regression Model
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)

# STEP 5: Random Forest Model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)

# STEP 6: Model Evaluation Function
def evaluate_model(model_name, y_test, y_pred, y_prob):
    print(f"{model_name} Evaluation:")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score:  {f1_score(y_test, y_pred):.4f}")
    print(f"AUC-ROC:   {roc_auc_score(y_test, y_prob):.4f}")

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix (rows=actual, cols=predicted):")
    print(cm)
    print()

    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "auc_roc": round(roc_auc_score(y_test, y_prob), 4),
    }

# Prediction and Probability for Logistic Regression
lr_pred = lr.predict(X_test_scaled)
lr_prob = lr.predict_proba(X_test_scaled)[:, 1]

# Prediction and Probability for Random Forest Classifier
rf_pred = rf.predict(X_test_scaled)
rf_prob = rf.predict_proba(X_test_scaled)[:, 1]

# Evaluate both base models
metrics = {}
metrics["logistic_regression"] = evaluate_model("Logistic Regression", y_test, lr_pred, lr_prob)
metrics["random_forest_base"] = evaluate_model("Random Forest", y_test, rf_pred, rf_prob)

# STEP 7: Hyperparameter Tuning for Random Forest
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5]
}
grid = GridSearchCV(
    RandomForestClassifier(random_state=42, class_weight="balanced"),
    param_grid,
    cv=5,
    scoring="recall"
)
grid.fit(X_train_scaled, y_train)
best_rf = grid.best_estimator_
print("Best Random Forest parameters:", grid.best_params_)

best_prob = best_rf.predict_proba(X_test_scaled)[:, 1]
best_pred = (best_prob >= 0.25).astype(int)
metrics["random_forest_tuned"] = evaluate_model("Random Forest (Tuned)", y_test, best_pred, best_prob)
metrics["best_rf_params"] = grid.best_params_

# STEP 8: Second model - predict Performance Rating
perf_features = ["Education", "JobInvolvement", "JobLevel", "MonthlyIncome", "YearsAtCompany", "YearsInCurrentRole"]

# Performance rating Feature and Target
X_perf = df[perf_features]
y_perf = df["PerformanceRating"].map({
    3: 0,
    4: 1
})

# Train Test Split
X_perf_train, X_perf_test, y_perf_train, y_perf_test = train_test_split(X_perf, y_perf, test_size=0.2, random_state=42, stratify=y_perf)

# Decision Tree Classifier
perf_model = DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42)
perf_model.fit(X_perf_train, y_perf_train)

perf_pred = perf_model.predict(X_perf_test)
perf_prob = perf_model.predict_proba(X_perf_test)[:, 1]

# Evaluation
metrics["performance_rating"] = evaluate_model("Performance Rating (Decision Tree)", y_perf_test, perf_pred, perf_prob)

# STEP 9: Saving Model files with joblib
joblib.dump(best_rf, "rf_model.pkl")
joblib.dump(lr, "lr_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X.columns.tolist(), "model_columns.pkl")
joblib.dump(perf_model, "perf_model.pkl")
joblib.dump(perf_features, "perf_features.pkl")

# Saving cleaned dataset
df.to_csv("employee_attrition_cleaned.csv", index=False)

print("Completed")