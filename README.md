# Santander Customer Prediction

This project demonstrates an end-to-end solution for predicting customer behavior using the LightGBM model. 

## Steps:
1. **Model Training**: Trained a LightGBM model for binary classification. Saved the model as pickle file.
2. **Streamlit UI**: A simple UI to upload test data and visualize predictions.
3. **FastAPI**: A REST API for real-time predictions.

Start the FastAPI server using:

```uvicorn api.main:app --reload```

To run the model:

```streamlit run app.py```

