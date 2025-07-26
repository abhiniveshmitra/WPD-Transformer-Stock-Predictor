# Stock Prediction Model with WPD and Transformer

This repository contains a robust Python implementation for stock price prediction. The approach combines **Wavelet Packet Decomposition (WPD)** for advanced feature engineering and a **Transformer neural network** for modeling sequential patterns in stock returns. The framework also includes comprehensive backtesting to evaluate trading strategies against standard financial benchmarks.

---

## Features

* **Data Acquisition**: Fetches historical stock prices via `yfinance` and macroeconomic indicators from FRED using `fredapi`.

* **Leak-Proof Data Preparation**: Computes all technical indicators and target variables within the bounds of training, validation, and testing splits to ensure no future data leakage.

* **Wavelet Packet Decomposition (WPD)**: Captures multi-frequency, non-stationary patterns in financial time series, enhancing predictive features beyond basic technical indicators.

* **Transformer Neural Network**: Utilizes a state-of-the-art deep learning model for sequence-to-sequence and sequence-to-one forecasting, specifically tuned for financial prediction.

* **Comprehensive Backtesting**: Evaluates the trading strategy with key metrics such as **Sharpe Ratio**, **Annualized Returns**, **Annualized Volatility**, and **Directional Accuracy**. Provides comparison against a buy-and-hold benchmark.

* **Automated Testing**: Supports batch evaluation across a basket of stocks, including sector-wise breakdowns and summary performance reports.

---

## Getting Started

### Prerequisites

* Python 3.8 or later
* FRED API key (for macroeconomic data). [Get one here.](https://fred.stlouisfed.org/docs/api/api_key.html)

### Installation

1. **Clone the repository:**

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

2. **Set up a virtual environment and install dependencies:**

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**`requirements.txt` includes:**

```
pandas
numpy
matplotlib
yfinance
torch
scikit-learn
fredapi
tqdm
pywavelets
```

### Configuration

* Open `stock_predictor.py` and replace the placeholder with your FRED API key:

```python
FRED_API_KEY = 'YOUR_FRED_API_KEY_HERE'  # Replace with your FRED API Key
```

> **Security Note:** For public codebases or production, use environment variables to manage API keys.

---

## Running the Script

You can run the model either for a **single stock** or for **automated batch testing**.

### 1. Single Stock Pipeline

Uncomment the `run_pipeline` call (and comment out `automate_testing`) at the end of `stock_predictor.py`:

```python
if __name__ == '__main__':
    print("\n" + "#"*50)
    print("### Running single stock pipeline (MSFT) ###")
    print("#"*50 + "\n")
    run_pipeline(
        ticker='MSFT',      # Change to your desired stock ticker
        period='2y',
        seq_length=40,
        epochs=50,
        batch_size=32,
        lr=0.001,
        print_backtest=True # Set False to skip plot
    )
    # automate_testing(...)
```

### 2. Automated Testing (Basket of Stocks)

Uncomment the `automate_testing` call (and comment out `run_pipeline`) at the end of `stock_predictor.py`. Customize `sector_stocks` in the function as needed.

```python
if __name__ == '__main__':
    print("\n" + "#"*50)
    print("### Running automated testing on a basket of stocks ###")
    print("#"*50 + "\n")
    automate_testing(
        period='2y',
        seq_length=40,
        epochs=50,
        batch_size=32,
        lr=0.001
    )
    # run_pipeline(...)
```

#### Run the script

```bash
python stock_predictor.py
```

---

## Performance Considerations

* **This pipeline is computationally intensive** due to deep learning and extensive feature engineering/backtesting. Running on a GPU is highly recommended for practical training times.

**Recommended Environments:**

* [Google Colab](https://colab.research.google.com/) (Free GPU)
* [Kaggle Notebooks](https://www.kaggle.com/code) (Free GPU/TPU)
* Cloud: AWS, GCP, Azure (GPU instances)
* Local machine with NVIDIA GPU (CUDA)

*Running entirely on CPU is possible, but will be slow, especially for batch/automated testing across many stocks.*

---


