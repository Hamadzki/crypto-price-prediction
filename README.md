# 📈 Crypto Price Analysis and Prediction

## 🌟 Overview
This repository contains a comprehensive project on predicting cryptocurrency prices using time series analysis and the `fbprophet` model. The project includes exploratory data analysis (EDA), model building for various cryptocurrencies, and performance evaluation. Additionally, a web app is developed to predict cryptocurrency prices based on user-uploaded data.

## 📂 Project Structure
- **📊 accuracy_graph/**: Contains graphs that depict model accuracy for each cryptocurrency.
- **🧠 models/**: Stores the saved `fbprophet` models for each cryptocurrency.
- **📈 Crypto_Bitcoin.ipynb**: Notebook for Bitcoin price prediction.
- **📈 Crypto_Chainlink.ipynb**: Notebook for Chainlink price prediction.
- **📈 Crypto_Polygon.ipynb**: Notebook for Polygon price prediction.
- **📈 Crypto_Theta_Network.ipynb**: Notebook for Theta Network price prediction.
- **📈 Crypto_xrp.ipynb**: Notebook for XRP price prediction.
- **🔍 EDA.ipynb**: Conducts exploratory data analysis of the cryptocurrency dataset.
- **🚀 app.py**: The main script for running the web app to predict cryptocurrency prices.
- **📄 crypto_dataset.csv**: The original cryptocurrency dataset used for analysis and model building.
- **📋 models_score.csv**: Initial model performance scores across different cryptocurrencies.
- **📋 models_score_final.csv**: Final model performance scores after tuning.
- **📝 requirements.txt**: Lists the Python dependencies required to run the project.

## 🌐 Web App
The web app allows users to upload cryptocurrency data files and predict future prices using the trained models. It is built using `Streamlit` and can be run locally or deployed on cloud services.

- **🚀 app.py**: The main script for deploying and running the web app.

## 📁 Data
- **📄 crypto_dataset.csv**: The historical cryptocurrency prices dataset used for model training.
- **📊 models_score.csv**: Initial scores for model accuracy.
- **📊 models_score_final.csv**: Final scores for model accuracy after tuning.

## 🤖 Models
- **📦 models/**: Contains saved models for each cryptocurrency.

## 🚀 Getting Started
1. Clone the repository:
   ```bash
   git clone https://github.com/hamadzki/crypto-price-prediction.git
   cd crypto-price-prediction

