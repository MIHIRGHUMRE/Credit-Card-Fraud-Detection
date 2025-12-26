import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load the trained model
@st.cache_resource
def load_model():
    with open('fraud_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

# Page Configuration
st.set_page_config(page_title="FraudGuard AI", page_icon="🛡️", layout="wide")

# Custom CSS for UI Styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        width: 100%;
        background-color: #007BFF;
        color: white;
    }
    .fraud {
        color: #d9534f;
        font-weight: bold;
        font-size: 24px;
    }
    .legit {
        color: #5cb85c;
        font-weight: bold;
        font-size: 24px;
    }
    </style>
    """, unsafe_allow_html=True)

# App Header
st.title("🛡️ FraudGuard: Credit Card Fraud Detection System")
st.markdown("Use Machine Learning to detect fraudulent credit card transactions based on anonymized transaction features.")

# Sidebar for Input Options
st.sidebar.header("Input Options")
input_method = st.sidebar.radio("Choose input method:", ["Use Example Data", "Upload CSV File", "Manual Entry (Advanced)"])

# Feature names (V1-V28 + Time + Amount)
feature_cols = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']

input_data = None

# --- METHOD 1: EXAMPLE DATA ---
if input_method == "Use Example Data":
    st.subheader("Select a Test Case")
    st.info("Since the dataset uses PCA features (V1-V28), manual entry is difficult. Use these pre-loaded examples to test the model.")
    
    # Pre-defined examples (taken from dataset averages/samples)
    example_type = st.radio("Choose transaction type:", ["Legitimate Transaction", "Fraudulent Transaction"])
    
    if example_type == "Legitimate Transaction":
        # A sample legit row from the dataset
        input_data = np.array([0.0, -1.359807, -0.072781, 2.536347, 1.378155, -0.338321, 0.462388, 0.239599, 
                               0.098698, 0.363787, 0.090794, -0.551600, -0.617801, -0.991390, -0.311169, 
                               1.468177, -0.470401, 0.207971, 0.025791, 0.403993, 0.251412, -0.018307, 
                               0.277838, -0.110474, 0.066928, 0.128539, -0.189115, 0.133558, -0.021053, 149.62])
    else:
        # A sample fraud row from the dataset
        input_data = np.array([472.0, -3.043541, -3.157307, 1.088463, 2.288644, 1.359805, -1.064823, 0.325574, 
                               -0.067794, -0.270953, -0.838587, -0.414575, -0.503141, 0.676502, -1.692029, 
                               2.000635, 0.666780, 0.599717, 1.725321, 0.283345, 2.102339, 0.661696, 
                               0.435477, 1.375966, -0.293803, 0.279798, -0.145362, -0.252773, 0.035764, 529.00])

    input_df = pd.DataFrame([input_data], columns=feature_cols)
    st.write("Input Data Preview:")
    st.dataframe(input_df)

# --- METHOD 2: UPLOAD CSV ---
elif input_method == "Upload CSV File":
    st.subheader("Batch Prediction")
    uploaded_file = st.file_uploader("Upload a CSV file (must contain columns Time, V1...V28, Amount)", type="csv")
    
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
        # Ensure only necessary columns are selected
        try:
            input_df = input_df[feature_cols]
            st.write("Uploaded Data:")
            st.dataframe(input_df.head())
            input_data = input_df.values
        except KeyError:
            st.error("The uploaded CSV is missing one or more required columns.")
            input_data = None

# --- METHOD 3: MANUAL ENTRY ---
elif input_method == "Manual Entry (Advanced)":
    st.subheader("Manual Feature Entry")
    # A text area to paste comma-separated values
    input_str = st.text_area("Paste comma-separated values (Time, V1...V28, Amount)", height=100)
    
    if input_str:
        try:
            input_list = [float(x.strip()) for x in input_str.split(',')]
            if len(input_list) == 30:
                input_data = np.array(input_list)
                input_df = pd.DataFrame([input_data], columns=feature_cols)
                st.write("Parsed Data:")
                st.dataframe(input_df)
            else:
                st.error(f"Expected 30 values, got {len(input_list)}.")
        except ValueError:
            st.error("Invalid format. Please ensure all values are numbers separated by commas.")

# --- PREDICTION LOGIC ---
if st.button("Analyze Transaction"):
    if input_data is not None:
        # If input is a single row (1D array)
        if input_data.ndim == 1:
            prediction = model.predict(input_data.reshape(1, -1))
            
            st.markdown("---")
            if prediction[0] == 0:
                st.markdown('<p class="legit">✅ Legitimate Transaction</p>', unsafe_allow_html=True)
                st.success("This transaction appears safe.")
            else:
                st.markdown('<p class="fraud">⚠️ Fraudulent Transaction Detected</p>', unsafe_allow_html=True)
                st.error("This transaction shows patterns associated with fraud.")
        
        # If input is batch (DataFrame/2D array)
        else:
            predictions = model.predict(input_data)
            input_df['Prediction'] = ["Fraud" if x == 1 else "Legit" for x in predictions]
            st.markdown("---")
            st.write("Batch Prediction Results:")
            st.dataframe(input_df.style.applymap(lambda v: 'color: red' if v == 'Fraud' else 'color: green', subset=['Prediction']))
            
            fraud_count = np.sum(predictions)
            st.warning(f"Detected {fraud_count} fraudulent transactions out of {len(predictions)}.")
    else:
        st.warning("Please provide input data first.")