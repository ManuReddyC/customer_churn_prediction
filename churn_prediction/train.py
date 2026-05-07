import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings

warnings.filterwarnings('ignore')

def load_and_clean_data(file_path="Telco-Customer-Churn.csv"):
    df = pd.read_csv(file_path)
    
    # Drop customerID
    df = df.drop('customerID', axis=1)
    
    # Fix TotalCharges
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(0, inplace=True)
    
    # Label encode binary categoricals
    le = LabelEncoder()
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
    for col in binary_cols:
        df[col] = le.fit_transform(df[col])
        
    return df

def feature_engineering(df):
    # tenure_group bins
    bins = [0, 12, 24, 36, 48, 60, 72]
    labels = [1, 2, 3, 4, 5, 6]
    df['tenure_group'] = pd.cut(df['tenure'], bins=bins, labels=labels, right=True, include_lowest=True)
    df['tenure_group'] = df['tenure_group'].astype(int)
    
    # avg_charge_per_month
    df['avg_charge_per_month'] = np.where(df['tenure'] > 0, df['TotalCharges'] / df['tenure'], df['MonthlyCharges'])
    
    # services_count
    services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    df['services_count'] = df[services].apply(lambda x: (x == 'Yes').sum(), axis=1)
    
    # has_tech_support binary
    df['has_tech_support'] = (df['TechSupport'] == 'Yes').astype(int)
    
    return df

def encode_categoricals(df):
    # One-hot encode multi-class categoricals
    cat_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
                'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
                'Contract', 'PaymentMethod']
    
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df

def main():
    print("Loading data...")
    df = load_and_clean_data()
    
    print("Feature engineering...")
    df = feature_engineering(df)
    
    print("Encoding categoricals...")
    df = encode_categoricals(df)
    
    # Ensure all column names are strings and clean for xgboost/lightgbm
    df.columns = [str(c).replace(' ', '_').replace('(', '').replace(')', '') for c in df.columns]
    
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    X.fillna(0, inplace=True)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Handling imbalance with SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    print("Training models...")
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        'LightGBM': LGBMClassifier(random_state=42)
    }
    
    results = []
    best_model = None
    best_auc = 0
    
    # Setup plot figures
    plt.figure(figsize=(10, 8))
    roc_ax = plt.gca()
    
    plt.figure(figsize=(10, 8))
    pr_ax = plt.gca()
    
    for name, model in models.items():
        model.fit(X_train_res, y_train_res)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        auc = roc_auc_score(y_test, y_proba)
        f1 = f1_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        
        results.append({
            'Model': name,
            'ROC-AUC': auc,
            'F1': f1,
            'Precision': prec,
            'Recall': rec
        })
        
        # Plot ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_ax.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')
        
        # Plot PR curve
        precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba)
        pr_ax.plot(recall_vals, precision_vals, label=f'{name}')
        
        if auc > best_auc:
            best_auc = auc
            best_model = model
            best_name = name
            best_y_pred = y_pred

    # Finish ROC Plot
    roc_ax.plot([0, 1], [0, 1], 'k--')
    roc_ax.set_xlabel('False Positive Rate')
    roc_ax.set_ylabel('True Positive Rate')
    roc_ax.set_title('ROC Curves')
    roc_ax.legend()
    roc_ax.figure.savefig('roc_curves.png')
    
    # Finish PR Plot
    pr_ax.set_xlabel('Recall')
    pr_ax.set_ylabel('Precision')
    pr_ax.set_title('Precision-Recall Curves')
    pr_ax.legend()
    pr_ax.figure.savefig('pr_curves.png')
    
    # Confusion Matrix for best model
    cm = confusion_matrix(y_test, best_y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Churn', 'Churn'])
    disp.plot(cmap='Blues')
    plt.title(f'Confusion Matrix ({best_name})')
    plt.savefig('confusion_matrix.png')
    
    print("\nEvaluation Results:")
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    print("\nSaving best model and features list...")
    # Save model and columns
    joblib.dump(best_model, 'best_model.pkl')
    joblib.dump(list(X.columns), 'model_features.pkl')
    print("Pipeline complete.")

if __name__ == "__main__":
    main()
