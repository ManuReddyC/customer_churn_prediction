import streamlit as st
import pandas as pd
import joblib
import requests
import shap
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
    }
    h1 {
        color: #00E5FF !important;
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        text-align: center;
        margin-bottom: 2rem;
    }
    h2, h3 {
        color: #E2E8F0 !important;
        font-family: 'Inter', sans-serif;
    }
    .stButton>button {
        background: linear-gradient(135deg, #00E5FF 0%, #007BFF 100%);
        color: #000;
        font-weight: bold;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0, 229, 255, 0.4);
        color: #fff;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.5rem;
        backdrop-filter: blur(10px);
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        background: -webkit-linear-gradient(#00E5FF, #007BFF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label {
        font-size: 1rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("✨ Customer Churn Risk & Retention Dashboard")

@st.cache_resource
def load_model():
    try:
        model = joblib.load('best_model.pkl')
        features = joblib.load('model_features.pkl')
        return model, features
    except Exception as e:
        return None, None

model, features = load_model()

if not model:
    st.error("Model not found. Please run the training pipeline first.")
    st.stop()

@st.cache_data
def load_data():
    df = pd.read_csv("Telco-Customer-Churn.csv")
    return df

df = load_data()

st.sidebar.header("Actions")
if st.sidebar.button("Score All Customers"):
    with st.spinner("Scoring customers..."):
        # We simulate batch scoring by applying preprocessing to the whole dataframe
        # For a real scenario, we might call the FastAPI endpoint for each, 
        # but here it's faster to score directly since we have the model loaded.
        
        # We need to preprocess the entire dataset
        # We'll use a slightly adapted preprocess logic
        score_data = df.copy()
        
        # Clean TotalCharges
        score_data['TotalCharges'] = pd.to_numeric(score_data['TotalCharges'], errors='coerce').fillna(0)
        
        # Apply preprocess_input to each row or just transform the whole DF
        # To do it quickly, we'll transform the whole DF using the train logic
        mapping = {'No': 0, 'Yes': 1, 'Female': 0, 'Male': 1}
        binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
        for col in binary_cols:
            score_data[col] = score_data[col].map(mapping).fillna(0)
            
        # Feature Engineering
        bins = [0, 12, 24, 36, 48, 60, 72]
        labels = [1, 2, 3, 4, 5, 6]
        score_data['tenure_group'] = pd.cut(score_data['tenure'], bins=bins, labels=labels, right=True, include_lowest=True).astype(int)
        
        import numpy as np
        score_data['avg_charge_per_month'] = np.where(score_data['tenure'] > 0, score_data['TotalCharges'] / score_data['tenure'], score_data['MonthlyCharges'])
        
        services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
        score_data['services_count'] = score_data[services].apply(lambda x: (x == 'Yes').sum(), axis=1)
        score_data['has_tech_support'] = (score_data['TechSupport'] == 'Yes').astype(int)
        
        cat_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
                    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
                    'Contract', 'PaymentMethod']
        
        score_data = pd.get_dummies(score_data, columns=cat_cols)
        score_data.columns = [str(c).replace(' ', '_').replace('(', '').replace(')', '') for c in score_data.columns]
        
        for f in features:
            if f not in score_data.columns:
                score_data[f] = False
                
        X_score = score_data[features].astype(float)
        
        # Predict
        probs = model.predict_proba(X_score)[:, 1]
        
        # Combine
        results_df = df.copy()
        results_df['Churn Probability'] = probs
        
        # Ensure TotalCharges is numeric for the calculation
        results_df['TotalCharges'] = pd.to_numeric(results_df['TotalCharges'], errors='coerce').fillna(0)
        
        # Calculate Risk x Revenue (using MonthlyCharges as revenue indicator for retention priority)
        results_df['Retention Priority Score'] = results_df['Churn Probability'] * results_df['MonthlyCharges']
        
        # Calculate summary metrics
        total_customers = len(results_df)
        high_risk_count = len(results_df[results_df['Churn Probability'] > 0.7])
        avg_churn_prob = results_df['Churn Probability'].mean() * 100
        revenue_at_risk = results_df[results_df['Churn Probability'] > 0.5]['MonthlyCharges'].sum()

        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; gap: 1rem;">
            <div class="metric-card" style="flex: 1;">
                <div class="metric-label">Total Evaluated</div>
                <div class="metric-value">{total_customers}</div>
            </div>
            <div class="metric-card" style="flex: 1;">
                <div class="metric-label">High Risk (>70%)</div>
                <div class="metric-value">{high_risk_count}</div>
            </div>
            <div class="metric-card" style="flex: 1;">
                <div class="metric-label">Avg Churn Prob</div>
                <div class="metric-value">{avg_churn_prob:.1f}%</div>
            </div>
            <div class="metric-card" style="flex: 1;">
                <div class="metric-label">Revenue at Risk / Mo</div>
                <div class="metric-value">${revenue_at_risk:,.0f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Sort top 200
        top_200 = results_df.sort_values(by='Retention Priority Score', ascending=False).head(200)
        
        st.subheader("⚠️ Top 200 At-Risk Customers (Priority = Prob * Monthly Revenue)")
        
        # Use Streamlit dataframe styling
        st.dataframe(
            top_200[['customerID', 'Churn Probability', 'MonthlyCharges', 'Retention Priority Score', 'Contract', 'tenure']]
            .style.background_gradient(subset=['Retention Priority Score', 'Churn Probability'], cmap='YlOrRd')
            .format({'Churn Probability': '{:.1%}', 'MonthlyCharges': '${:.2f}', 'Retention Priority Score': '{:.2f}'}),
            use_container_width=True,
            height=400
        )
        
        st.subheader("🔍 SHAP Explanation for Top Customer")
        top_customer = top_200.iloc[0]
        st.write(f"**Customer ID:** `{top_customer['customerID']}` | **Probability:** `{top_customer['Churn Probability']:.1%}` | **Revenue:** `${top_customer['MonthlyCharges']:.2f}`")
        
        # Explainer
        idx = top_200.index[0]
        customer_features = X_score.iloc[[idx]]
        
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if type(model).__name__ in ['RandomForestClassifier', 'XGBClassifier', 'LGBMClassifier', 'GradientBoostingClassifier']:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer(customer_features)
                if isinstance(shap_values, list):
                    shap_obj = shap_values[1][0]
                elif len(shap_values.shape) == 3:
                    shap_obj = shap_values[0, :, 1]
                else:
                    shap_obj = shap_values[0]
            else:
                masker = shap.maskers.Independent(data=X_score, max_samples=100)
                predict_fn = lambda x: model.predict_proba(x)[:, 1]
                explainer = shap.Explainer(predict_fn, masker)
                shap_values = explainer(customer_features)
                shap_obj = shap_values[0]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        # Style the plot with dark theme compatible colors if needed, but SHAP defaults are usually ok.
        plt.style.use('dark_background')
        shap.plots.waterfall(shap_obj, show=False)
        st.pyplot(fig)

st.write("👈 Click **'Score All Customers'** in the sidebar to generate the retention priority list.")
