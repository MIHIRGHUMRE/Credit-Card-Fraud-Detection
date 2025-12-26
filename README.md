# 🛡️Credit Card Fraud Detection

A Machine Learning application deployed on Streamlit that detects fraudulent credit card transactions.

## 📊 Project Overview
This project analyzes a dataset of European credit card transactions to build a classifier capable of distinguishing fraud from legitimate payments. 

* **Dataset**: Contains PCA-transformed features (V1-V28), Time, and Amount.
* **Imbalance Handling**: Used Under-Sampling to balance the dataset (492 Fraud vs 492 Legit).
* **Model**: Logistic Regression (Selected for efficiency and high recall).
* **Accuracy**: ~94% on Test Data.

## 📂 Repository Structure
* `Project_Credit_Card_Fraud_Detection.ipynb`: The research notebook containing EDA, training logic, and evaluation.
* `app.py`: The user-facing Streamlit application.
* `fraud_model.pkl`: The trained model saved via Pickle.
* `requirements.txt`: List of Python dependencies.

## 🚀 How to Run Locally

1.  **Clone the Repo**:
    ```bash
    git clone [https://github.com/YOUR_USERNAME/FraudGuard-AI.git](https://github.com/YOUR_USERNAME/FraudGuard-AI.git)
    cd FraudGuard-AI
    ```
2.  **Install Requirements**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the App**:
    ```bash
    streamlit run app.py
    ```

## 🧠 Model Workflow
1.  **Data Loading**: Load transaction CSV.
2.  **Preprocessing**: Scale 'Amount' and handle class imbalance.
3.  **Training**: Train Logistic Regression on balanced data.
4.  **Evaluation**: Check Accuracy, Precision, and Recall.
5.  **Deployment**: Serialize model and load into Streamlit.

---

## 🌐 Live Demo
You can access the live version of the application here:  
👉 **https://credit-card-fraud-detection-eetartth4dadh6ojkw4owx.streamlit.app/** 
