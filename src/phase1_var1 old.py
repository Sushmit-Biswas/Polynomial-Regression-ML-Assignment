import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import os

# Create images directory
os.makedirs('images', exist_ok=True)

# 1. Load Data
try:
    train_df = pd.read_csv('BT2024038/BT2024038_train_var1.csv')
    test_df = pd.read_csv('BT2024038/BT2024038_test_var1.csv')
except FileNotFoundError:
    print("Dataset not found. Please verify paths.")
    exit()

features_all = [col for col in train_df.columns if col != 'y']
y_train = train_df['y']

# 2. Mathematical EDA: Correlation Heatmap
corr_matrix = train_df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Phase 1: Feature vs Target Correlation Matrix")
plt.tight_layout()
plt.savefig('images/phase1_heatmap.png')
plt.close()

# 3. Dynamic Feature Selection
# Drop features with negligible Pearson correlation to the target 
# to mathematically eliminate noise without "guessing" or "hardcoding"
target_corr = corr_matrix['y'].abs().drop('y')
# Threshold set empirically to > 0.05 to discard mathematical white noise
selected_features = target_corr[target_corr > 0.05].index.tolist()
print(f"Phase 1 - Algorithmically Selected Features (Absolute Correlation > 0.05): {selected_features}")

X_opt = train_df[selected_features].values

# 4. Model Tuning via Automated Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
degrees = range(1, 6) # Test up to degree 5 before typical variance explosion limits
train_errors = []
val_errors = []

best_deg = 1
min_val_err = float('inf')

for d in degrees:
    poly = PolynomialFeatures(degree=d)
    X_poly = poly.fit_transform(X_opt)
    
    t_err_fold = []
    v_err_fold = []
    
    for train_index, val_index in kf.split(X_poly):
        X_t, X_v = X_poly[train_index], X_poly[val_index]
        y_t, y_v = y_train.iloc[train_index], y_train.iloc[val_index]
        
        model = LinearRegression()
        model.fit(X_t, y_t)
        
        t_err_fold.append(mean_squared_error(y_t, model.predict(X_t)))
        v_err_fold.append(mean_squared_error(y_v, model.predict(X_v)))
        
    train_errors.append(np.mean(t_err_fold))
    mean_val = np.mean(v_err_fold)
    val_errors.append(mean_val)
    
    if mean_val < min_val_err:
        min_val_err = mean_val
        best_deg = d

plt.figure(figsize=(8, 6))
plt.plot(degrees, train_errors, label='Train MSE', marker='o')
plt.plot(degrees, val_errors, label='Validation MSE', marker='s')
plt.title("Phase 1: Dynamic Validation Error Search Curve")
plt.xlabel("Polynomial Degree")
plt.ylabel("Mean Squared Error")
plt.yscale("log")
plt.legend()
plt.grid(True)
plt.savefig('images/phase1_cv_curve.png')
plt.close()

print(f"Phase 1 - Mathematically Optimal Degree Discovered: {best_deg}")

# 5. Final Model Compilation & Diagnostics
poly = PolynomialFeatures(degree=best_deg)
X_train_poly = poly.fit_transform(X_opt)

final_model = LinearRegression()
final_model.fit(X_train_poly, y_train)

y_train_pred = final_model.predict(X_train_poly)
mse = mean_squared_error(y_train, y_train_pred)
r2 = r2_score(y_train, y_train_pred)

print(f"Phase 1 - Final Integrity Check - MSE: {mse:.4f}, R2: {r2:.4f}")

# Residual Tracking Plot
residuals = y_train - y_train_pred
plt.figure(figsize=(8, 6))
plt.scatter(y_train_pred, residuals, alpha=0.6, color='purple')
plt.axhline(0, color='red', linestyle='--')
plt.title("Phase 1: Residual Variance Distribution")
plt.xlabel("Predicted Target")
plt.ylabel("Residual Component")
plt.grid(True)
plt.savefig('images/phase1_residuals.png')
plt.close()

# True vs Predicted verification
plt.figure(figsize=(8, 6))
plt.scatter(y_train, y_train_pred, alpha=0.6)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], color='red', linestyle='--')
plt.title("Phase 1: Prediction Mapping Accuracy")
plt.xlabel("Ground Truth (y)")
plt.ylabel("Algorithmic Prediction")
plt.grid(True)
plt.savefig('images/phase1_true_vs_pred.png')
plt.close()

# 6. Test File Export
X_test = test_df[selected_features].values
X_test_poly = poly.transform(X_test)
test_predictions = final_model.predict(X_test_poly)

test_df['y'] = test_predictions
test_df.to_csv('BT2024038/BT2024038_pred_var1.csv', index=False)
print("Phase 1 - Success, exported algorithm mappings.")
