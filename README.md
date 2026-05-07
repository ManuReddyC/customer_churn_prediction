# Customer Churn Prediction Project

This project predicts customer churn using various machine learning models (Logistic Regression, Random Forest, XGBoost, LightGBM), handles class imbalance with SMOTE, provides SHAP explanations, exposes a FastAPI for inference, and includes a Streamlit dashboard.

## Setup

1. The dependencies are listed in `requirements.txt`. Install them using:
   ```bash
   pip install -r requirements.txt
   ```

2. To train the models and generate the `best_model.pkl` along with performance plots (`roc_curves.png`, `pr_curves.png`, `confusion_matrix.png`):
   ```bash
   python train.py
   ```

3. To start the FastAPI server and Streamlit dashboard together, you can run:
   ```cmd
   .\start.bat
   ```
   Or separately:
   - FastAPI: `uvicorn api:app --host 0.0.0.0 --port 8000`
   - Streamlit: `streamlit run dashboard.py`

## Features

- **train.py**: Runs the data cleaning, feature engineering, and model training loop. It evaluates multiple classifiers and saves the best one.
- **api.py**: Provides a `/predict` endpoint to score individual customers.
- **dashboard.py**: A web app that simulates batch scoring for retention prioritization, and displays SHAP values for the most at-risk customer.
