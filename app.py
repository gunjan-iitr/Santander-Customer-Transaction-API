import streamlit as st
import pandas as pd
import requests
from io import StringIO

# Streamlit UI
st.title("Santander Customer Prediction")
st.markdown("Upload a CSV file to get prediction results from the model.")

# File uploader
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded CSV file into a pandas DataFrame
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.write(df.head())  # Display the first few rows of the uploaded file

    # Show the user uploaded data preview
    st.markdown("### API Predictions:")
    st.markdown("Once you upload a CSV file, the results will appear here.")

    # Button to trigger API prediction
    if st.button("Get Predictions"):
        # Prepare the file for API (Streamlit reads files in memory, so we need to convert it to CSV string)
        data = StringIO()
        df.to_csv(data, index=False)
        data.seek(0)

        # Send data to the FastAPI /predict endpoint
        api_url = "http://127.0.0.1:8000/predict"  # FastAPI local endpoint
        response = requests.post(api_url, files={"file": data})

        # Handle API response
        if response.status_code == 200:
            result = response.json()
            roc_auc = result['roc_auc']
            classification_report = result['classification_report']

            # Display results
            st.subheader(f"📈 Model ROC-AUC: {roc_auc}")
            st.subheader("📊 Classification Report:")
            st.json(classification_report)
        else:
            st.error("Error: Could not get predictions from the API.")
