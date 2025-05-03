# Fraud Detection Models Comparitive Analysis

This repository contains an exploratory data analysis notebook and a collection of machine learning models tested for performance benchmarking and comparison.

## Repository Structure

```
.
├── EDA.ipynb                         # Exploratory Data Analysis notebook
└── final_models/
    ├── 1 - Logistic Regression + GA.ipynb
    ├── 2 - Naïve Bayes.ipynb
    ├── 3 - Isolation Forest with Experimental Scenarios.ipynb
    ├── 4 - Xgboost.ipynb
    ├── 5 - Random Forest Classifier + SMOTE ENN.ipynb
    ├── 6_Artificial_Neural_Network.ipynb
    ├── 7 - CNN.ipynb
    ├── 8-LightGBM.ipynb
    ├── 9 - XGBOD.ipynb
    ├── 10 - Hybrid Classifier(Logistic_Regression,_Decision_Tree,_Random_Forest).ipynb
    └── 11 - SVM.ipynb
    └── 12 - RUS + XGBoost.ipynb
```

## How to Run

1. **Clone the Repository**

   ```bash
   git clone https://github.com/aadhamashraf/Fraud-Detection-Interpretability-and-Explainability.git
   cd Fraud-Detection-Interpretability-and-Explainability
   ```

2. **Launch Jupyter Notebook**

   ```bash
   jupyter notebook
   ```

3. **Notebook Execution Order**

   * Start with `EDA.ipynb` for data understanding.
   * Navigate to the `final_models` folder to explore and run individual model notebooks:

     * Classical models (e.g., Logistic Regression, Naïve Bayes)
     * Ensemble models (e.g., XGBoost, LightGBM, Random Forest)
     * Deep learning models (ANN, CNN)
     * Hybrid and experimental approaches (e.g., XGBOD, Hybrid Classifier, Isolation Forest)