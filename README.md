# Fraud detection in financial transactions using explainable ML & DL models 

This repository contains the final project for the DSAI 305 course at Zewail City, focusing on explainable AI models for financial transaction fraud detection. The `final_models/` directory includes Jupyter notebooks, each implementing a machine learning or deep learning model with explainability techniques.

## Instructions
1. **Open Notebooks**:
   - Navigate to the `final_models/` directory.
   - Open each Jupyter notebook (e.g., `LightGBM_Model.ipynb`, `CNN_Model.ipynb`) in a Jupyter environment (e.g., JupyterLab, Google Colab).

2. **Run All Cells**:
   - Execute cells sequentially, as data, imports and dependencies are defined at the top.
   - Each notebook covers data preprocessing, model training, evaluation, and explainability, so review the Table of Contents.


## Notes
- Ensure dependencies (`pandas`, `numpy`, `scikit-learn`, `imblearn`, `lightgbm`, `tensorflow`, `shap`, `lime`) are installed via `pip`.
- Datasets are loaded within notebooks (e.g., PaySim via `kagglehub`). (Imported automatically in each notebook).
- For resource-intensive explainability sections (e.g., SHAP, LIME), adjust sample sizes if needed.
