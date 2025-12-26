import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="FraudGuard AI", 
    page_icon="🛡️", 
    layout="wide"
)

# --- CUSTOM CSS FOR PROFESSIONAL UI ---
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #007BFF;
        color: white;
        height: 3em;
        font-weight: bold;
    }
    .fraud-text {
        color: #d9534f;
        font-weight: bold;
        font-size: 20px;
    }
    .legit-text {
        color: #28a745;
        font-weight: bold;
        font-size: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    try:
        # Ensure fraud_model.pkl exists in the same directory
        with open('fraud_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("❌ Model file 'fraud_model.pkl' not found. Please run your notebook to generate it.")
        return None

model = load_model()

# --- APP HEADER ---
st.title("🛡️ FraudGuard: Credit Card Fraud Detection")
st.markdown("Developed with Machine Learning to identify suspicious financial transactions.")
st.write("---")

# --- SIDEBAR INPUT ---
st.sidebar.header("Navigation & Input")
input_method = st.sidebar.radio(
    "Choose Input Method:", 
    ["Example Scenarios", "Batch Upload (CSV)", "Manual Entry"]
)

# Feature list for reference
feature_cols = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
input_df = None

# --- INPUT LOGIC ---
if input_method == "Example Scenarios":
    st.subheader("Quick Test Scenarios")
    scenario = st.selectbox(
        "Select a transaction type to simulate:",
        ["Standard Legitimate Transaction", "High-Risk Fraudulent Transaction"]
    )
    
    if scenario == "Standard Legitimate Transaction":
        # Data sample for a legit transaction
        data = np.array([0.0, -1.35, -0.07, 2.53, 1.37, -0.33, 0.46, 0.23, 0.09, 0.36, 0.09, -0.55, -0.61, -0.99, -0.31, 1.46, -0.47, 0.20, 0.02, 0.40, 0.25, -0.01, 0.27, -0.11, 0.06, 0.12, -0.18, 0.13, -0.02, 149.62])
    else:
        # Data sample for a fraud transaction
        data = np.array([472.0, -3.04, -3.15, 1.08, 2.28, 1.35, -1.06, 0.32, -0.06, -0.27, -0.83, -0.41, -0.50, 0.67, -1.69, 2.00, 0.66, 0.59, 1.72, 0.28, 2.10, 0.66, 0.43, 1.37, -0.29, 0.27, -0.14, -0.25, 0.03, 529.00])
    
    input_df = pd.DataFrame([data], columns=feature_cols)
    st.write("Selected Transaction Data:")
    st.dataframe(input_df)

elif input_method == "Batch Upload (CSV)":
    st.subheader("Batch File Processing")
    uploaded_file = st.file_uploader("Upload transaction CSV (must have 30 columns: Time, V1-V28, Amount)", type="csv")
    if uploaded_file:
        input_df = pd.read_csv(uploaded_file)
        # Ensure we only use the necessary columns in correct order
        input_df = input_df[feature_cols]
        st.success(f"Successfully loaded {len(input_df)} transactions.")
        st.dataframe(input_df.head())

elif input_method == "Manual Entry":
    st.subheader("Manual Transaction Entry")
    st.info("Paste 30 comma-separated values below (Time, V1...V28, Amount).")
    raw_input = st.text_area("Input Values:", placeholder="0.0, -1.35, ...")
    if raw_input:
        try:
            val_list = [float(x.strip()) for x in raw_input.split(',')]
            if len(val_list) == 30:
                input_df = pd.DataFrame([val_list], columns=feature_cols)
                st.dataframe(input_df)
            else:
                st.error(f"Error: Expected 30 values, but received {len(val_list)}.")
        except ValueError:
            st.error("Error: Please enter only numeric values separated by commas.")

# --- PREDICTION AND RESULTS ---
st.write("---")
if st.button("RUN FRAUD ANALYSIS"):
    if input_df is not None and model is not None:
        # Get predictions
        predictions = model.predict(input_df.values)
        input_df['Status'] = ["Fraud" if x == 1 else "Legit" for x in predictions]
        
        # 1. Show Metrics
        total = len(predictions)
        frauds = int(np.sum(predictions))
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Scanned", total)
        col2.metric("Fraud Detected", frauds, delta=frauds, delta_color="inverse")
        col3.metric("System Status", "Active", delta_color="normal")
        
        st.write("---")
        
        # 2. Results Handling
        if total == 1:
            # Single Transaction UI
            if predictions[0] == 0:
                st.markdown('<p class="legit-text">✅ Result: Transaction is LEGITIMATE</p>', unsafe_allow_html=True)
                st.balloons()
            else:
                st.markdown('<p class="fraud-text">⚠️ Warning: Transaction is FRAUDULENT</p>', unsafe_allow_html=True)
        else:
            # Batch Results UI (The Option 1 Fix)
            st.subheader("Detailed Analysis Report")
            
            if frauds > 0:
                st.error(f"Action Required: {frauds} suspicious transactions flagged.")
                # Filter and style ONLY the fraud rows
                fraud_data = input_df[input_df['Status'] == "Fraud"]
                st.write("🚩 Flagged Transactions (Fraud Only):")
                # We style only the small fraud subset to avoid the rendering error
                st.dataframe(fraud_data.style.applymap(lambda v: 'color: red; font-weight: bold', subset=['Status']))
            else:
                st.success("Analysis complete. No fraudulent patterns detected in this batch.")

            # Show all data WITHOUT styling to ensure performance and avoid cell limits
            with st.expander("View Full Transaction Log (Unstyled)"):
                st.write("Complete dataset with predictions:")
                st.dataframe(input_df)
                
            # Allow user to download results
            csv = input_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Full Report as CSV", csv, "fraud_report.csv", "text/csv")
            
    else:
        st.warning("Please provide transaction data to perform analysis.")

# --- FOOTER ---
st.write("---")
st.caption("FraudGuard AI v1.0 | Machine Learning Portfolio Project")
