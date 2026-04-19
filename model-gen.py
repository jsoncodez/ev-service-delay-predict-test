import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, classification_report, mean_absolute_error, r2_score, confusion_matrix
from sklearn.inspection import PartialDependenceDisplay
# import shap
from sklearn.preprocessing import LabelEncoder

# Dataset load from csv to data frame
df = pd.read_csv("./data/car_service_dataset.csv")

# Encode Categorical datatypes to num tool
le = LabelEncoder()
df['service_type'] = le.fit_transform(df['service_type'])

# Features of dataset
features = [
    'month', 'day_of_week', 'is_weekend', 'service_type', 'avg_service_time',
    'appointments_per_day', 'service_slots', 'num_technicians', 'backlog_size',
    'demand_capacity_ratio'
]

X = df[features]
# targets:
y_class = df['delay_risk']  # classification target
y_reg = df['wait_time']  # regression target

# Train-Test
X_train, X_test, y_train_class, y_test_class = train_test_split(X, y_class, test_size=0.2, random_state=42)
_, _, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)


####### Classification Model - Delay Risk Prediction ########

# Logistic Regression (base)
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train_class)

y_pred_log = log_model.predict(X_test)


# Evaluation Metrics for Logistic Regression
print("Logistic Regression:")
print("Accuracy:", accuracy_score(y_test_class, y_pred_log))
print("Recall:", recall_score(y_test_class, y_pred_log))
print("Precision:", precision_score(y_test_class, y_pred_log))
print(classification_report(y_test_class, y_pred_log))

# Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train_class)
1
y_pred_rf = rf_model.predict(X_test)


# Evaluation Metrics for Random Forest Classifier
print("Random Forest Classifier:")
print("Accuracy:", accuracy_score(y_test_class, y_pred_rf))
print("Recall:", recall_score(y_test_class, y_pred_rf))
print("Precision:", precision_score(y_test_class, y_pred_rf))
print(classification_report(y_test_class, y_pred_rf))

# Confusion Matrix for Random Forest
sns.heatmap(confusion_matrix(y_test_class, y_pred_rf), annot=True, fmt='d')
plt.title("Delay Prediction Confusion Matrix")
plt.show()

# Feature Importance for Random Forest
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title("Feature Importance - Delay Prediction")
plt.show()

# Partial Dependence Plot for Delay Risk
PartialDependenceDisplay.from_estimator(rf_model, X_test, ['demand_capacity_ratio'])
plt.show()


joblib.dump(rf_model, './models/rf_model_classification_delay_risk.pkl')


####### Regression Model - Wait Time Prediction ########


# Linear Regression (baseline)
lin_model = LinearRegression()
lin_model.fit(X_train, y_train_reg)

y_pred_lin = lin_model.predict(X_test)

# Evaluation Metrics for Linear Regression
print("Linear Regression:")
print("MAE:", mean_absolute_error(y_test_reg, y_pred_lin))
print("R2:", r2_score(y_test_reg, y_pred_lin))

# Random Forest Regressor
rf_reg = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
rf_reg.fit(X_train, y_train_reg)

y_pred_rf_reg = rf_reg.predict(X_test)


# Evaluation Metrics for Random Forest Regressor
print("Random Forest Regressor:")
print("MAE:", mean_absolute_error(y_test_reg, y_pred_rf_reg))
print("R2:", r2_score(y_test_reg, y_pred_rf_reg))

# Feature Importance for Random Forest Regressor
feature_importance_reg = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_reg.feature_importances_
}).sort_values(by='Importance', ascending=False)

sns.barplot(x='Importance', y='Feature', data=feature_importance_reg)
plt.title("Feature Importance - Wait Time Prediction")
plt.show()

# Partial Dependence Plot for Wait Time
PartialDependenceDisplay.from_estimator(rf_reg, X_test, ['backlog_size'])
plt.show()

joblib.dump(log_model, './models/rf_model_regression_wait_time.pkl')