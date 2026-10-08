from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures


ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PRED_DIR = ROOT / "predictions"
PRED_DIR.mkdir(exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR = ROOT / "images" / "p2"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

def main():
    train_df = pd.read_csv(RAW_DIR / "BT2024038_train_var2.csv")
    test_df = pd.read_csv(RAW_DIR / "BT2024038_test_var2.csv")
    features = [column for column in train_df.columns if column != "y"]
    X = train_df[features].to_numpy()
    y = train_df["y"].to_numpy()

    pairplot = sns.pairplot(train_df)
    pairplot.figure.savefig(IMAGE_DIR / "phase2_pairplot.png", dpi=150)
    plt.close(pairplot.figure)

    plt.figure(figsize=(6, 5))
    sns.heatmap(train_df.corr(numeric_only=True), annot=True, cmap="viridis", fmt=".2f")
    plt.title("Phase 2: Feature and target correlations")
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "phase2_heatmap.png", dpi=150)
    plt.close()

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    degrees = range(1, 21)
    train_mse = []
    validation_mse = []
    best_degree = None
    best_validation_mse = np.inf

    for degree in degrees:
        transformer = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = transformer.fit_transform(X)
        fold_train_errors = []
        fold_validation_errors = []

        for train_index, validation_index in kfold.split(X_poly):
            model = LinearRegression()
            model.fit(X_poly[train_index], y[train_index])
            fold_train_errors.append(
                mean_squared_error(y[train_index], model.predict(X_poly[train_index]))
            )
            fold_validation_errors.append(
                mean_squared_error(y[validation_index], model.predict(X_poly[validation_index]))
            )

        train_mse.append(np.mean(fold_train_errors))
        validation_mse.append(np.mean(fold_validation_errors))
        if validation_mse[-1] < best_validation_mse:
            best_validation_mse = validation_mse[-1]
            best_degree = degree

    plt.figure(figsize=(8, 6))
    plt.plot(list(degrees), train_mse, marker="o", label="Cross-validation train MSE")
    plt.plot(list(degrees), validation_mse, marker="s", label="Cross-validation validation MSE")
    plt.xlabel("Polynomial degree")
    plt.ylabel("Mean squared error")
    plt.title("Phase 2: Degree selection by 5-fold cross-validation")
    plt.yscale("log")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "phase2_cv_curve.png", dpi=150)
    plt.close()

    transformer = PolynomialFeatures(degree=best_degree, include_bias=False)
    X_poly = transformer.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)
    y_pred = model.predict(X_poly)

    oof_predictions = np.empty_like(y, dtype=float)
    for train_index, validation_index in kfold.split(X):
        fold_transformer = PolynomialFeatures(degree=best_degree, include_bias=False)
        X_train_poly = fold_transformer.fit_transform(X[train_index])
        X_validation_poly = fold_transformer.transform(X[validation_index])
        fold_model = LinearRegression().fit(X_train_poly, y[train_index])
        oof_predictions[validation_index] = fold_model.predict(X_validation_poly)

    print(f"Phase 2 features: {features}")
    print(f"Phase 2 selected degree: {best_degree}")
    print(f"Phase 2 training MSE: {mean_squared_error(y, y_pred):.6f}")
    print(f"Phase 2 training R2: {r2_score(y, y_pred):.6f}")
    print(f"Phase 2 OOF MSE: {mean_squared_error(y, oof_predictions):.6f}")
    print(f"Phase 2 OOF R2: {r2_score(y, oof_predictions):.6f}")

    residuals = y - y_pred
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.6, color="darkorange")
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Fitted prediction")
    plt.ylabel("Residual (actual - fitted)")
    plt.title("Phase 2: Residuals versus fitted values")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "phase2_residuals.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.scatter(y, y_pred, alpha=0.6, color="teal")
    limits = [min(y.min(), y_pred.min()), max(y.max(), y_pred.max())]
    plt.plot(limits, limits, "r--", label="Ideal prediction")
    plt.xlabel("Actual y")
    plt.ylabel("Fitted y")
    plt.title("Phase 2: Actual versus fitted values")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "phase2_true_vs_pred.png", dpi=150)
    plt.close()

    test_predictions = model.predict(transformer.transform(test_df[features].to_numpy()))
    pd.DataFrame({"y": test_predictions}).to_csv(
        PRED_DIR / "BT2024038_pred_var2.csv", index=False
    )

if __name__ == "__main__":
    main()
