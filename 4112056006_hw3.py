import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
import statsmodels.api as sm

# Set style for aesthetics
plt.style.use('ggplot')
sns.set_palette("husl")

# 1. Load Data
file_path = 'winequality-red.csv'
df = pd.read_csv(file_path)

# --- [CRISP-DM: Data Understanding] ---
print("--- Data Understanding ---")
print(df.info())

# Correlation analysis for feature selection
plt.figure(figsize=(10, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.close()

# --- [CRISP-DM: Data Preparation] ---
# Separate features and target
X = df.drop('quality', axis=1)
y = df['quality']

# Feature Selection: Select Top 8 features based on f_regression
selector = SelectKBest(score_func=f_regression, k=8)
X_new = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()]
print(f"\nSelected Features: {list(selected_features)}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- [CRISP-DM: Modeling] ---
# Simple Linear Regression with Scikit-learn
model = LinearRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# Statsmodels for detailed summary and Confidence Intervals
X_train_sm = sm.add_constant(X_train_scaled)
sm_model = sm.OLS(y_train, X_train_sm).fit()

# --- [CRISP-DM: Evaluation] ---
print("\n--- Model Evaluation ---")
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")
print(f"R-squared: {r2:.4f}")

# Plot 1: Actual vs Predicted with Prediction Interval
plt.figure(figsize=(10, 6))
sns.regplot(x=y_test, y=y_pred, ci=95, line_kws={"color": "blue"}, scatter_kws={"alpha": 0.5})
plt.xlabel('Actual Quality')
plt.ylabel('Predicted Quality')
plt.title('Actual vs Predicted Wine Quality (with 95% CI)')
plt.tight_layout()
plt.savefig('prediction_plot.png')
plt.close()

# Plot 2: Residuals
plt.figure(figsize=(10, 6))
residuals = y_test - y_pred
sns.histplot(residuals, kde=True, color='purple')
plt.title('Residuals Distribution')
plt.tight_layout()
plt.savefig('residuals_plot.png')
plt.close()

print("\nAnalysis Complete. Plots saved as 'correlation_heatmap.png', 'prediction_plot.png', and 'residuals_plot.png'.")
print("\nStatsmodels Summary:")
print(sm_model.summary())
