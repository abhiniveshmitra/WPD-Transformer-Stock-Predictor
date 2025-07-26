Stock Prediction Model with WPD and Transformer
This repository contains a Python script that implements a stock price prediction model. The model leverages Wavelet Packet Decomposition (WPD) for advanced feature engineering and a Transformer neural network for sequence prediction. It also includes a comprehensive backtesting framework to evaluate the strategy's performance.

Features
Data Acquisition: Fetches historical stock data using yfinance and macroeconomic indicators from FRED using fredapi.

Leak-Proof Data Preparation: Ensures no future information leakage by computing technical indicators and target variables separately for training, validation, and testing splits.

Wavelet Packet Decomposition (WPD): Transforms features to capture multi-frequency patterns in financial time series, enhancing the model's ability to learn complex relationships.

Transformer Neural Network: A powerful deep learning architecture designed for sequence-to-sequence or sequence-to-one tasks, adapted here for predicting future stock returns.

Comprehensive Backtesting: Evaluates the trading strategy's performance using key financial metrics such as Sharpe Ratio, Annualized Returns, Annualized Volatility, and Directional Accuracy, comparing it against a simple buy-and-hold benchmark.

Automated Testing: Includes functionality to run the prediction and backtesting pipeline across a predefined basket of stocks, providing overall and sector-wise performance summaries.

Getting Started
Prerequisites
Before running the script, ensure you have Python 3.8+ installed. You'll also need a FRED API key to fetch macroeconomic data. You can obtain one from the FRED website.

Installation
Clone the repository:

git clone [https://github.com/abhiniveshmitra/WPD-Transformer-Stock-Predictor.git](https://github.com/abhiniveshmitra/WPD-Transformer-Stock-Predictor)
cd WPD-Transformer-Stock-Predictor

(Replace YOUR_USERNAME and YOUR_REPO_NAME with your actual GitHub username and repository name.)

Install dependencies:
It's highly recommended to use a virtual environment.

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
pip install -r requirements.txt

requirements.txt content:

pandas
numpy
matplotlib
yfinance
torch
scikit-learn
fredapi
tqdm
pywavelets

Configuration
Open stock_predictor.py and replace 'YOUR_FRED_API_KEY_HERE' with your actual FRED API key:

FRED_API_KEY = 'YOUR_FRED_API_KEY_HERE' # Replace with your FRED API Key

Security Note: For production environments or if you plan to share your code publicly, it's highly recommended to manage API keys using environment variables rather than hardcoding them directly in the script.

Running the Script
You can run the script to either test a single stock or automate testing across a basket of stocks.

Run for a single stock (e.g., MSFT):
Uncomment the run_pipeline call and comment out the automate_testing call at the end of stock_predictor.py. You can modify the ticker and other parameters as needed.

if __name__ == '__main__':
    # Example of running the pipeline for a single stock
    print("\n" + "#"*50)
    print("### Running single stock pipeline (MSFT) ###")
    print("#"*50 + "\n")
    run_pipeline(
        ticker='MSFT', # Change the ticker here to any stock symbol
        period='2y',
        seq_length=40,
        epochs=50,
        batch_size=32,
        lr=0.001,
        print_backtest=True # Set to False if you don't want the individual plot
    )
    # comment out the automate_testing call
    # automate_testing(...)

Run automated testing on a basket of stocks:
Ensure the automate_testing call is uncommented and the run_pipeline call is commented out at the end of stock_predictor.py. You can customize the sector_stocks dictionary within the automate_testing function to define your desired basket of stocks.

if __name__ == '__main__':
    # Example of running the automated testing on a basket of stocks
    print("\n" + "#"*50)
    print("### Running automated testing on a basket of stocks ###")
    print("#"*50 + "\n")
    automate_testing(
        period='2y',      # Data period for each stock
        seq_length=40,    # Sequence length for Transformer input
        epochs=50,        # Training epochs for each model
        batch_size=32,    # Batch size for training
        lr=0.001          # Learning rate
    )
    # comment out the run_pipeline call
    # run_pipeline(...)

To execute the script:

python stock_predictor.py

Performance Considerations
This model involves deep learning (Transformer) and extensive data processing (WPD, feature engineering, backtesting across multiple stocks). These operations are computationally intensive.

It is highly recommended to run this script in a GPU-accelerated environment for practical execution times. Options include:

Google Colab: A free cloud-based Jupyter notebook environment that provides access to GPUs.

Kaggle Notebooks: Similar to Colab, offering free GPU/TPU access.

Cloud Platforms: AWS, Google Cloud Platform (GCP), Azure, etc., which offer GPU instances.

Local Machine with GPU: If you have a compatible NVIDIA GPU and CUDA installed.

Running the script on a CPU will work, but it will be significantly slower, especially when using the automate_testing function across many stocks.
