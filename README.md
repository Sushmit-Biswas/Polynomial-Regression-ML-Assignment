# Polynomial Regression ML Assignment

This repository dictates a robust, zero-regularization polynomial regression pipeline evaluating non-linear covariates and minimizing mathematical variance mathematically without violating rigid bounds. 

## Structure 
- `/data/raw/` -> The original, un-modified Train and Test datasets (Target `y` mapping variations).
- `/predictions/` -> The computationally evaluated exact prediction output arrays mapped mathematically across the theoretical bounds without header indexing.
- `/assignment_docs/` -> The original problem statement constraints and the generated theoretical final latex reports.
- `/images/p1/` and `/images/p2/` -> Automatically regenerated out-of-fold metrics tracing variance and MSE optimization loops separated by Phase.
- `/src/` -> The python models executing the regression architectures.
- `requirements.txt` & `README.md` are bound securely at the root hierarchy.

## Run Protocol
Scripts natively isolate their operational constraints mathematically through local dependencies relative to `../`.
```bash
pip install -r requirements.txt

# Run Steam Turbine Variant (Phase 1)
cd src
python phase1_var1.py

# Run Geological Thermal Variant (Phase 2)
python phase2_var2.py
```
