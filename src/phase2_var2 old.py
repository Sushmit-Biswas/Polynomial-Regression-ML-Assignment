import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import os
import warnings
warnings.filterwarnings('ignore') # Filter standard scikit-learn ill-conditioned matrix warnings during high-degree tests

os.makedirs('images', exist_ok=True)

# 1. Load Assignment Data
try:
    train_df = pd.read_csv('BT2024038/BT2024038_train_var2.csv')
    test_df = pd.read_csv('BT2024038/BT2024038_test_var2.csv')
except FileNotFoundError:
    print("Dataset not found. Please verify.")
    exit()

# 2. EDA Phase
sns.pairplot(train_df)
plt.savefig('images/phase2_pairplot.png')
plt.close()

corr_matrix = train_df.corr()
plt.figure(figsize=(6, 5))
sns.heatmap(corr_matrix, annot=True, cmap='viridis', fmt=".2f")
plt.title("Phase 2: Mathematical Correlation Matrix")
plt.tight_layout()
plt.savefig('images/phase2_heatmap.png')
plt.close()

# 3. Unbiased Feature Elimination 
# Algorithmically eliminating uncorrelated features (White Noise variables generating 0 covariance)
target_corr = corr_matrix['y'].abs().drop('y')
selected_features = target_corr[target_corr > 0.05].index.tolist()
print(f"Phase 2 - Algorithmically Selected Features (Absolute Correlation > 0.05): {selected_features}")

X_opt = train_df[selected_features].values
y_train = train_df['y']

# 4. CV Complexity Search
kf = KFold(n_splits=5, shuffle=True, random_state=42)
degrees = range(1, 10) 
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
    mean_val_err = np.mean(v_err_fold)
    val_errors.append(mean_val_err)
    
    if mean_val_err < min_val_err:
        min_val_err = mean_val_err
        best_deg = d

plt.figure(figsize=(8, 6))
plt.plot(degrees, train_errors, label='Train MSE', marker='o')
plt.plot(degrees, val_errors, label='Validation MSE', marker='s')
plt.title("Phase 2: K-Fold Polynomial Bound Curve")
plt.xlabel("Polynomial Complexity Degree")
plt.ylabel("MSE")
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.savefig('images/phase2_cv_curve.png')
plt.close()

print(f"Phase 2 - Mathematically Optimal Degree Discovered: {best_deg}")

# 5. Final Model Architecture
poly = PolynomialFeatures(degree=best_deg)
X_train_opt_poly = poly.fit_transform(X_opt)

final_model = LinearRegression()
final_model.fit(X_train_opt_poly, y_train)

y_train_pred = final_model.predict(X_train_opt_poly)
mse = mean_squared_error(y_train, y_train_pred)
r2 = r2_score(y_train, y_train_pred)
print(f"Phase 2 - Final Model Verification - MSE: {mse:.4f}, R2: {r2:.4f}")

# Residuals plotting
residuals = y_train - y_train_pred
plt.figure(figsize=(8, 6))
plt.scatter(y_train_pred, residuals, alpha=0.6, color='darkorange')
plt.axhline(0, color='red', linestyle='--')
plt.title("Phase 2: Error Distribution Plot")
plt.xlabel("Polynomial Prediction")
plt.ylabel("Calculated Deviation")
plt.grid(True)
plt.savefig('images/phase2_residuals.png')
plt.close()

# 6. Evaluation Dump
X_test = test_df[selected_features].values
X_test_poly = poly.transform(X_test)
test_predictions = final_model.predict(X_test_poly)

test_df['y'] = test_predictions
test_df.to_csv('BT2024038/BT2024038_pred_var2.csv', index=False)
print("Phase 2 - Success, exported algorithm mappings.")
