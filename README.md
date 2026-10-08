# Polynomial Regression ML Assignment

This repository contains the code and deliverables for a polynomial regression machine learning assignment. It uses cross-validation to find the best polynomial degree for two different datasets without using regularization (like Ridge or Lasso).

## Structure 
- `/data/raw/` -> The original Train and Test CSV datasets.
- `/predictions/` -> The final CSV prediction files containing the generated `y` values.
- `/docs/` -> The original problem statement PDF and the final LaTeX report files.
- `/images/phase1/` and `/images/phase2/` -> The cross-validation, heatmap, and residual plot images for Phase 1 and Phase 2.
- `/src/` -> The Python scripts that run the machine learning models.
- `requirements.txt` & `README.md` -> Standard project configuration files at the root level.

## How to Run
Make sure to install the required libraries first, then run the scripts from inside the `src` folder:
```bash
pip install -r requirements.txt

# Run Phase 1
cd src
python phase1_var1.py

# Run Phase 2
python phase2_var2.py
```
