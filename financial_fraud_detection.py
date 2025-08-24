# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Set page configuration
st.set_page_config(
    page_title="Financial Fraud Detection",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("💰 Financial Fraud Detection System")
st.markdown("""
This application detects fraudulent financial transactions using machine learning.
Upload your transaction data or use the sample data to analyze for potential fraud.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Select a page:", 
                          ["Data Overview", "Exploratory Analysis", "Fraud Detection", "Model Performance"])

# Load data function with caching
@st.cache_data
def load_data():
    try:
        data = pd.read_csv("Synthetic_Financial_datasets_log.csv")
        return data
    except FileNotFoundError:
        st.error("Sample data file not found. Please upload your own data.")
        return None

# Load the data
data = load_data()

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload your own CSV file", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

if data is None:
    st.stop()

# Data Overview Page
if options == "Data Overview":
    st.header("Data Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("First 5 Rows")
        st.dataframe(data.head())
    
    with col2:
        st.subheader("Dataset Information")
        st.text(f"Shape: {data.shape}")
        st.text(f"Columns: {', '.join(data.columns)}")
        
        # Data types
        dtype_df = pd.DataFrame(data.dtypes, columns=['Data Type'])
        st.dataframe(dtype_df)
    
    st.subheader("Missing Values")
    missing_df = pd.DataFrame(data.isnull().sum(), columns=['Missing Values'])
    st.dataframe(missing_df)
    
    st.subheader("Fraud Distribution")
    col1, col2 = st.columns(2)
    
    with col1:
        fraud_counts = data['isFraud'].value_counts()
        st.dataframe(fraud_counts)
        
        fraud_percent = (data['isFraud'].sum() / len(data)) * 100
        st.metric("Fraud Percentage", f"{fraud_percent:.2f}%")
    
    with col2:
        fig, ax = plt.subplots()
        fraud_counts.plot(kind='bar', ax=ax, color=['green', 'red'])
        ax.set_title("Fraud vs Non-Fraud Transactions")
        ax.set_xlabel("isFraud")
        ax.set_ylabel("Count")
        st.pyplot(fig)

# Exploratory Analysis Page
elif options == "Exploratory Analysis":
    st.header("Exploratory Data Analysis")
    
    # Transaction types
    st.subheader("Transaction Types Distribution")
    type_counts = data['type'].value_counts()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.dataframe(type_counts)
    
    with col2:
        fig, ax = plt.subplots()
        type_counts.plot(kind='bar', color='orange', ax=ax)
        ax.set_title("Different Transaction Types")
        ax.set_xlabel("Type")
        ax.set_ylabel("Count")
        st.pyplot(fig)
    
    # Fraud rate by transaction type
    st.subheader("Fraud Rate by Transaction Type")
    mean_fraud_type = data.groupby('type')['isFraud'].mean()
    
    fig, ax = plt.subplots()
    mean_fraud_type.plot(kind='bar', ax=ax)
    ax.set_title("Fraud Rate by Transaction Type")
    ax.set_ylabel("Fraud Rate")
    st.pyplot(fig)
    
    # Amount analysis
    st.subheader("Transaction Amount Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.dataframe(data['amount'].describe())
    
    with col2:
        fig, ax = plt.subplots()
        ax.hist(data['amount'], bins=100, color='purple')
        ax.set_xlabel("Transaction Amount")
        ax.set_ylabel("Frequency")
        ax.set_title("Amount Distribution")
        st.pyplot(fig)
    
    # Boxplot for amount vs fraud
    st.subheader("Amount vs Fraud (for amounts < 50,000)")
    fig, ax = plt.subplots()
    sns.boxplot(data=data[data['amount'] < 50000], x='isFraud', y='amount', ax=ax)
    ax.set_title("Amount vs Fraud")
    st.pyplot(fig)
    
    # Correlation matrix
    st.subheader("Correlation Heatmap")
    corr_matrix = data[['amount', 'oldbalanceOrg', 'newbalanceOrig', 
                       'oldbalanceDest', 'newbalanceDest', 'isFraud']].corr()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)

# Fraud Detection Page
elif options == "Fraud Detection":
    st.header("Fraud Detection")
    
    # Preprocess data for ML
    df_ml = data.drop(['nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1)
    df_ml.dropna(subset=['isFraud'], inplace=True)
    
    X = df_ml.drop('isFraud', axis=1)
    y = df_ml['isFraud']
    
    # Encoding
    X = pd.get_dummies(X, columns=['type'], drop_first=True)
    X = X.fillna(0)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    
    # Standardizing numeric columns
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model or load existing
    try:
        model = joblib.load("simple_fraud_model.pkl")
        st.success("Pre-trained model loaded successfully!")
    except:
        st.info("Training a new model... This may take a moment.")
        model = LogisticRegression(class_weight='balanced', max_iter=1000)
        model.fit(X_train_scaled, y_train)
        joblib.dump(model, "simple_fraud_model.pkl")
        st.success("Model trained and saved successfully!")
    
    # Make predictions
    preds = model.predict(X_test_scaled)
    pred_proba = model.predict_proba(X_test_scaled)
    
    # Display results
    st.subheader("Detection Results")
    
    # Show some example predictions
    results_df = pd.DataFrame({
        'Actual': y_test.values,
        'Predicted': preds,
        'Fraud Probability': pred_proba[:, 1]
    })
    
    st.dataframe(results_df.head(20))
    
    # Fraud probability distribution
    st.subheader("Fraud Probability Distribution")
    fig, ax = plt.subplots()
    ax.hist(pred_proba[:, 1], bins=50, color='red', alpha=0.7)
    ax.set_xlabel("Predicted Probability of Fraud")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Fraud Probabilities")
    st.pyplot(fig)
    
    # Download predictions
    results_csv = results_df.to_csv(index=False)
    st.download_button(
        label="Download Predictions as CSV",
        data=results_csv,
        file_name="fraud_predictions.csv",
        mime="text/csv"
    )

# Model Performance Page
elif options == "Model Performance":
    st.header("Model Performance Evaluation")
    
    # Preprocess data for ML
    df_ml = data.drop(['nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1)
    df_ml.dropna(subset=['isFraud'], inplace=True)
    
    X = df_ml.drop('isFraud', axis=1)
    y = df_ml['isFraud']
    
    # Encoding
    X = pd.get_dummies(X, columns=['type'], drop_first=True)
    X = X.fillna(0)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    
    # Standardizing numeric columns
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Load model
    try:
        model = joblib.load("simple_fraud_model.pkl")
    except:
        st.error("Model not found. Please train the model first on the Fraud Detection page.")
        st.stop()
    
    # Make predictions
    preds = model.predict(X_test_scaled)
    
    # Display metrics
    st.subheader("Performance Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        accuracy = accuracy_score(y_test, preds)
        st.metric("Accuracy", f"{accuracy*100:.2f}%")
    
    with col2:
        # Precision, Recall, F1 for the positive class (fraud)
        report = classification_report(y_test, preds, output_dict=True)
        precision = report['1']['precision']
        st.metric("Precision (Fraud)", f"{precision*100:.2f}%")
    
    with col3:
        recall = report['1']['recall']
        st.metric("Recall (Fraud)", f"{recall*100:.2f}%")
    
    # Confusion matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, preds)
    
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)
    
    # Classification report
    st.subheader("Detailed Classification Report")
    report_df = pd.DataFrame(classification_report(y_test, preds, output_dict=True)).transpose()
    st.dataframe(report_df)
    
    # Feature importance
    st.subheader("Feature Importance")
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.coef_[0]
    }).sort_values('importance', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(10), ax=ax)
    ax.set_title("Top 10 Most Important Features")
    st.pyplot(fig)

# Footer
st.sidebar.markdown("---")
st.sidebar.info(
    "This is a demo financial fraud detection application. "
    "For real-world use, please consult with financial security experts."
)