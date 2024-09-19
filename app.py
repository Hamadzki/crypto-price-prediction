import pandas as pd
import streamlit as st 
from datetime import datetime
import numpy as np
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
np.float_ = np.float64
from prophet import Prophet
import pickle


st.title("Crypto price prediction")

# st.image('image.jpg')

models_df = pd.read_csv('models_score_final.csv')

crypto = st.sidebar.selectbox('Select the currency', models_df['name'])

smape = float(models_df[models_df['name'] == crypto].reset_index()['r2_score'][0])

st.sidebar.write(f"{crypto} has the r2_score: {abs(smape)}")

if smape > 0.70:
    st.sidebar.write('This model is currently in development mode. Please select any other currency')

selected_date = st.sidebar.date_input("Select a date", datetime.today())
specific_date = pd.to_datetime(selected_date)

model_path = f'models/{crypto}.pkl'
model_scaler = f'models/{crypto}_scaler.pkl'
model_pt = f'models/{crypto}_pt.pkl'

model_image_path = f'accuracy_graph/{crypto}.png'

# st.image(model_image_path)
choice = st.sidebar.radio(f"Do you want to see the accuracy graph for {crypto} model", ['No', 'Yes'])
if choice == 'Yes':    
    st.image(model_image_path, caption="Model accuracy", use_column_width=True, width=1200)

df = pd.read_csv('crypto_dataset.csv')

if st.button('Predict'):
    # model = joblib.load(model_path)

    # Load the model using pickle
    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    with open(model_scaler, 'rb') as file:
        loaded_scaler = pickle.load(file)

    with open(model_pt, 'rb') as file:
        pt = pickle.load(file)

    future_df = pd.DataFrame({'ds': [specific_date]})
    forecast = model.predict(future_df)

 # Inverse Yeo-Johnson transformation (reshape to 2D before applying inverse_transform)
    forecast['yhat'] = pt.inverse_transform(forecast[['yhat']].values.reshape(-1, 1))
    forecast['yhat_lower'] = pt.inverse_transform(forecast[['yhat_lower']].values.reshape(-1, 1))
    forecast['yhat_upper'] = pt.inverse_transform(forecast[['yhat_upper']].values.reshape(-1, 1))

    # Inverse scaling (reshape is not required since inverse_transform of scalers handles it)
    forecast['yhat'] = loaded_scaler.inverse_transform(forecast[['yhat']])
    forecast['yhat_lower'] = loaded_scaler.inverse_transform(forecast[['yhat_lower']])
    forecast['yhat_upper'] = loaded_scaler.inverse_transform(forecast[['yhat_upper']])

    # Flatten the arrays back to 1D (optional, for easier use)
    forecast['yhat'] = forecast['yhat'].to_numpy().flatten()
    forecast['yhat_lower'] = forecast['yhat_lower'].to_numpy().flatten()
    forecast['yhat_upper'] = forecast['yhat_upper'].to_numpy().flatten()

    st.write(forecast['yhat'][0])
    pred = forecast['yhat'][0]
    lower = forecast['yhat_lower'][0]
    upper = forecast['yhat_upper'][0]
    st.write(f"Lower Limit: {lower:.2f}USD **<< Predicted Value: {pred:.2f}USD <<** Upper Limit: {upper:.2f}USD")

