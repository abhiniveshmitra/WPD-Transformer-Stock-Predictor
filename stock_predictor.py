# stock_predictor.py
#
# This script implements a stock price prediction model using a Wavelet Packet Decomposition (WPD)
# and a Transformer neural network. It fetches historical stock data and macroeconomic indicators,
# processes them, trains the model, and performs backtesting to evaluate the strategy's performance
# against a buy-and-hold benchmark.
#
# Key Features:
# - Data acquisition from Yahoo Finance (stock) and FRED (macroeconomic indicators).
# - Leak-proof data preparation: features and target are computed per train/validation/test split.
# - Wavelet Packet Decomposition (WPD) for feature transformation to capture multi-frequency patterns.
# - Transformer model for sequence prediction.
# - Comprehensive backtesting with financial metrics (Sharpe Ratio, Annualized Returns, Directional Accuracy).
# - Automated testing across a basket of stocks and sector-wise analysis.

# --- Library Imports ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from fredapi import Fred
from tqdm import tqdm
import pywt
import torch.nn.functional as F
import random
from sklearn.metrics import mean_squared_error, r2_score
import warnings
import os

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- IMPORTANT: Replace with your own FRED API Key ---
# You can obtain a FRED API key from: https://fred.stlouisfed.org/docs/api/api_key.html
# For production environments, consider using environment variables (e.g., os.environ.get('FRED_API_KEY'))
FRED_API_KEY = 'YOUR_FRED_API_KEY_HERE'
fred = Fred(api_key=FRED_API_KEY)

# Check if GPU is available and set the device for PyTorch operations
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# --- Data Acquisition Functions ---

def fetch_stock_data(ticker, period='2y'):
    """
    Fetches raw historical stock data for a given ticker.
    Does not compute indicators here; indicators are computed per split later to prevent lookahead bias.

    Args:
        ticker (str): The stock ticker symbol (e.g., 'AAPL', 'MSFT').
        period (str): The period for which to fetch data (e.g., '1y', '2y', '5y', 'max').

    Returns:
        pd.DataFrame: DataFrame containing historical stock data (Open, High, Low, Close, Volume).
    """
    print(f"Fetching stock data for {ticker} for period {period}...")
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    if df.empty:
        print(f"Warning: No stock data found for {ticker} with period {period}.")
    return df

def fetch_macro_data(start_date, end_date):
    """
    Fetches key macroeconomic indicators from FRED for a specific date range.
    Data is forward-filled within the specified range to handle missing values.

    Args:
        start_date (datetime): The start date for fetching macroeconomic data.
        end_date (datetime): The end date for fetching macroeconomic data.

    Returns:
        pd.DataFrame: DataFrame containing macroeconomic indicators.
    """
    print(f"Fetching macroeconomic data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
    indicators = {
        'DFF': 'Fed_Funds_Rate',    # Federal Funds Effective Rate
        'T10Y2Y': 'Yield_Curve',    # 10-Year Treasury Constant Maturity Minus 2-Year Treasury Constant Maturity
        'UNRATE': 'Unemployment_Rate', # Civilian Unemployment Rate
        'CPIAUCSL': 'CPI',          # Consumer Price Index for All Urban Consumers
        'VIXCLS': 'VIX',            # CBOE Volatility Index
    }
    # Create a date range as index to ensure all dates are covered
    macro_df = pd.DataFrame(index=pd.date_range(start=start_date, end=end_date))

    for code, name in indicators.items():
        try:
            series = fred.get_series(code, start_date, end_date)
            if not series.empty:
                macro_df[name] = series
        except Exception as e:
            print(f"Could not fetch {name} (FRED code: {code}): {e}")

    # Remove timezone information if present for consistent joining
    if not macro_df.empty and macro_df.index.tz:
        macro_df.index = macro_df.index.tz_localize(None)

    # Forward-fill missing macro data points within the fetched range
    macro_df = macro_df.ffill()
    return macro_df

def create_raw_combined_dataset(ticker, period='2y'):
    """
    Combines raw stock data and macroeconomic data into a single DataFrame.
    This function fetches raw data without any feature engineering or target creation.

    Args:
        ticker (str): The stock ticker symbol.
        period (str): The period for stock data.

    Returns:
        pd.DataFrame: Combined raw stock and macroeconomic data.
    """
    print(f"Preparing raw combined data for {ticker}...")
    stock_df = fetch_stock_data(ticker, period=period)

    if stock_df.empty:
        raise ValueError(f"Failed to fetch stock data for {ticker}. Cannot proceed.")

    # Ensure stock data index is timezone-naive for consistent merging
    if stock_df.index.tz:
        stock_df.index = stock_df.index.tz_localize(None)

    start_date = stock_df.index.min()
    end_date = stock_df.index.max()
    macro_df = fetch_macro_data(start_date, end_date)

    # Join raw dataframes based on date index
    # 'how=left' ensures all stock dates are kept, and macro data is aligned
    combined_df = stock_df.join(macro_df, how='left')
    # Forward-fill any remaining gaps in macro data after joining
    combined_df.ffill(inplace=True)
    # Drop rows with any remaining NaN values (e.g., at the very beginning if not enough data for FFILL)
    combined_df.dropna(inplace=True)

    if combined_df.empty:
        raise ValueError(f"Combined dataset for {ticker} is empty after merging and cleaning.")

    return combined_df


# --- Feature Engineering and Data Preparation ---

def compute_features_per_split(df_split):
    """
    Computes technical indicators, additional features, and the target variable for a given data split.
    This function is designed to be applied *after* data splitting to prevent lookahead bias.

    Args:
        df_split (pd.DataFrame): A subset of the raw combined data (e.g., train, validation, or test split).

    Returns:
        pd.DataFrame: DataFrame with computed features and the 'Target' variable.
    """
    # Ensure the input DataFrame is a copy to avoid modifying the original split
    df_split = df_split.copy()

    # Technical Indicators
    df_split['MA5'] = df_split['Close'].rolling(window=5).mean()
    df_split['MA20'] = df_split['Close'].rolling(window=20).mean()

    # Moving Average Convergence Divergence (MACD)
    df_split['EMA12'] = df_split['Close'].ewm(span=12, adjust=False).mean()
    df_split['EMA26'] = df_split['Close'].ewm(span=26, adjust=False).mean()
    df_split['MACD'] = df_split['EMA12'] - df_split['EMA26']
    df_split['Signal_Line'] = df_split['MACD'].ewm(span=9, adjust=False).mean()

    # Relative Strength Index (RSI)
    delta = df_split['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    # Handle division by zero for rs calculation
    rs = np.where(loss == 0, np.inf, gain / loss)
    df_split['RSI'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    df_split['BB_Middle'] = df_split['Close'].rolling(window=20).mean()
    df_split['BB_Std'] = df_split['Close'].rolling(window=20).std()
    df_split['BB_Upper'] = df_split['BB_Middle'] + 2 * df_split['BB_Std']
    df_split['BB_Lower'] = df_split['BB_Middle'] - 2 * df_split['BB_Std']

    # Additional Features
    df_split['Daily_Return'] = df_split['Close'].pct_change()
    df_split['Volatility'] = df_split['Daily_Return'].rolling(window=20).std()
    df_split['Momentum'] = df_split['Close'].pct_change(periods=5)

    # Target: Next day's return. Shift by -1 to get the future return.
    # The last row will become NaN and must be dropped.
    df_split['Target'] = df_split['Daily_Return'].shift(-1)

    # Drop rows with NaN values resulting from rolling window calculations or target shifting
    df_split = df_split.dropna()

    return df_split

def apply_wpd_to_data(data, wavelet='db3', level=2):
    """
    Applies Wavelet Packet Decomposition (WPD) to each feature in the dataset.
    WPD decomposes signals into different frequency bands, which can help capture
    multi-scale patterns in financial time series.

    Args:
        data (pd.DataFrame): DataFrame containing features and the 'Target' column.
        wavelet (str): The name of the wavelet to use (e.g., 'db3', 'haar').
        level (int): The decomposition level.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: WPD-transformed features.
            - np.ndarray: Target values.
    """
    target_col = 'Target'
    # Exclude 'Target' and any non-numeric columns from feature processing
    feature_cols = [col for col in data.columns if col != target_col and pd.api.types.is_numeric_dtype(data[col])]

    wpd_features = []
    for col in feature_cols:
        values = data[col].values
        # Perform Wavelet Packet Decomposition
        wp = pywt.WaveletPacket(data=values, wavelet=wavelet, mode='symmetric', maxlevel=level)
        # Get nodes at the specified level
        nodes = [node.path for node in wp.get_level(level, 'natural')]
        reconstructed_coeffs = []
        for node_path in nodes:
            # Reconstruct signal from specific packet coefficients
            new_wp = pywt.WaveletPacket(data=None, wavelet=wavelet, mode='symmetric', maxlevel=level)
            new_wp[node_path] = wp[node_path].data
            rec = new_wp.reconstruct(update=True)

            # Pad or truncate reconstructed signal to match original length
            if len(rec) < len(values):
                pad_width = len(values) - len(rec)
                rec = np.pad(rec, (0, pad_width), 'constant')
            elif len(rec) > len(values):
                rec = rec[:len(values)]
            reconstructed_coeffs.append(rec)

        # Stack reconstructed coefficients for each feature horizontally
        if reconstructed_coeffs:
            wpd_features.append(np.column_stack(reconstructed_coeffs))
        else:
            # If no WPD features generated (e.g., due to very short data), use original feature
            wpd_features.append(values.reshape(-1, 1))

    # Concatenate all WPD-transformed features across columns
    if not wpd_features:
        raise ValueError("No WPD features were generated. Check input data or WPD parameters.")
    combined_wpd_features = np.concatenate(wpd_features, axis=1)

    return combined_wpd_features, data[[target_col]].values

def prepare_sequences(combined_features, target, seq_length=40):
    """
    Creates sequences of features and corresponding targets for the Transformer model.
    Each sequence consists of `seq_length` consecutive time steps of features,
    and the target is the return at `seq_length` + 1.

    Args:
        combined_features (np.ndarray): WPD-transformed features.
        target (np.ndarray): Target values.
        seq_length (int): The length of each input sequence.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: X (sequences of features).
            - np.ndarray: y (corresponding target values).
    """
    X, y = [], []
    for i in range(len(combined_features) - seq_length):
        X.append(combined_features[i:i + seq_length])
        y.append(target[i + seq_length])
    return np.array(X), np.array(y)


# --- PyTorch Model Definition ---

class WPDTransformer(nn.Module):
    """
    A Transformer-based neural network model for time series prediction.
    It takes WPD-transformed sequences as input.

    Args:
        input_dim (int): The dimensionality of each time step in the input sequence (number of features).
        seq_length (int): The length of the input sequences.
        d_model (int): The dimension of the embedding space and the Transformer's internal representation.
        nhead (int): The number of attention heads in the Transformer.
        num_layers (int): The number of Transformer encoder layers.
        dropout (float): The dropout rate for regularization.
    """
    def __init__(self, input_dim, seq_length, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super(WPDTransformer, self).__init__()
        # Linear layer to embed input features into the d_model dimension
        self.embedding = nn.Linear(input_dim, d_model)
        self.seq_length = seq_length
        # Positional embeddings to inject temporal information into the sequences
        self.position_embeddings = nn.Embedding(seq_length, d_model)

        # Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2, # Dimension of the feedforward network model
            dropout=dropout,
            batch_first=True # Input and output tensors are (batch, sequence, feature)
        )
        # Stack multiple Transformer Encoder Layers
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Fully connected layers for the final prediction
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_model // 2, 1) # Output a single prediction (e.g., next day's return)

    def forward(self, x):
        """
        Forward pass through the Transformer model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_dim).

        Returns:
            torch.Tensor: Predicted output of shape (batch_size, 1).
        """
        batch_size, seq_len, _ = x.shape
        # Generate position IDs dynamically based on current sequence length
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=x.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len)

        # Apply linear embedding and add positional embeddings
        x = self.embedding(x)
        position_embeddings = self.position_embeddings(position_ids)
        x = x + position_embeddings

        # Pass through Transformer encoder
        x = self.transformer_encoder(x)

        # Use the last element of the sequence for prediction (common in sequence-to-one tasks)
        x = x[:, -1, :]
        x = F.relu(self.fc1(x)) # Apply ReLU activation to the first FC layer
        x = self.dropout1(x)    # Apply dropout
        x = self.fc2(x)         # Final prediction layer
        return x

class WPDDataset(Dataset):
    """
    Custom PyTorch Dataset for WPD-transformed stock data.
    """
    def __init__(self, X, y):
        """
        Initializes the dataset.

        Args:
            X (np.ndarray): Features (sequences).
            y (np.ndarray): Targets.
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.X)

    def __getitem__(self, idx):
        """
        Retrieves a sample from the dataset at the given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the feature sequence and its corresponding target.
        """
        return self.X[idx], self.y[idx]


# --- Model Training and Evaluation ---

def train_model(model, train_loader, val_loader, epochs=50, lr=0.001):
    """
    Handles the training loop for the Transformer model.
    Includes early stopping and learning rate scheduling.

    Args:
        model (nn.Module): The PyTorch model to train.
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        epochs (int): Number of training epochs.
        lr (float): Initial learning rate.

    Returns:
        nn.Module: The trained model (best model based on validation loss).
    """
    criterion = nn.MSELoss() # Mean Squared Error loss for regression
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5) # Adam optimizer with L2 regularization
    # Reduce learning rate when validation loss stops improving
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)

    best_val_loss = float('inf')
    early_stopping_patience = 10 # Number of epochs to wait for improvement before stopping
    early_stopping_counter = 0

    print("Starting model training...")
    for epoch in range(epochs):
        model.train() # Set model to training mode
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device) # Move data to device (CPU/GPU)
            optimizer.zero_grad() # Clear gradients from previous step
            y_pred = model(X_batch) # Forward pass
            loss = criterion(y_pred, y_batch) # Calculate loss
            if torch.isnan(loss):
                print(f"Warning: NaN loss encountered in training at epoch {epoch+1}. Skipping batch.")
                continue
            loss.backward() # Backpropagation
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Clip gradients to prevent exploding gradients
            optimizer.step() # Update model parameters
            train_loss += loss.item() # Accumulate training loss

        # Validation phase
        model.eval() # Set model to evaluation mode
        val_loss = 0
        with torch.no_grad(): # Disable gradient calculation for validation
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
                val_loss += criterion(y_pred, y_batch).item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        scheduler.step(val_loss) # Update learning rate based on validation loss

        if (epoch + 1) % 5 == 0: # Print progress every 5 epochs
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            torch.save(model.state_dict(), 'best_model.pt') # Save the best model
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print(f'Early stopping at epoch {epoch+1} due to no improvement in validation loss.')
                break

    print("Training complete. Loading best model for evaluation.")
    model.load_state_dict(torch.load('best_model.pt')) # Load the best saved model state
    return model

def evaluate_model(model, test_loader, scaler_y):
    """
    Evaluates the trained model on the test set and inverse transforms the predictions
    to the original scale.

    Args:
        model (nn.Module): The trained PyTorch model.
        test_loader (DataLoader): DataLoader for the test dataset.
        scaler_y (MinMaxScaler): The scaler used to transform the target variable.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: Inverse-transformed predictions.
            - np.ndarray: Inverse-transformed actual target values.
    """
    model.eval() # Set model to evaluation mode
    predictions, actuals = [], []
    with torch.no_grad(): # Disable gradient calculation
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_pred_scaled = model(X_batch)

            # Inverse transform predictions and actuals back to original scale
            y_pred = scaler_y.inverse_transform(y_pred_scaled.cpu().numpy())
            y_true = scaler_y.inverse_transform(y_batch.numpy())

            predictions.append(y_pred)
            actuals.append(y_true)

    predictions = np.concatenate(predictions)
    actuals = np.concatenate(actuals)

    return predictions, actuals


# --- Backtesting and Performance Analysis ---

def backtest_strategy(predictions, actuals, ticker=None, print_results=True):
    """
    Performs backtesting of the trading strategy and calculates key financial performance metrics.
    The strategy is to go long (buy) if the predicted return is positive, and short (sell) if negative.

    Args:
        predictions (np.ndarray): Model's predicted returns.
        actuals (np.ndarray): Actual observed returns.
        ticker (str, optional): Stock ticker symbol for plotting purposes. Defaults to None.
        print_results (bool): Whether to print detailed results and plot cumulative returns.

    Returns:
        dict: A dictionary containing various performance metrics.
    """
    if print_results:
        print("\n--- Backtesting Results ---")

    results = pd.DataFrame({
        'Actual_Return': actuals.flatten(),
        'Predicted_Return': predictions.flatten()
    })

    # Generate trading signal: +1 for predicted positive return, -1 for predicted negative return
    results['Signal'] = np.sign(results['Predicted_Return'])
    # Calculate strategy return: Signal * Actual_Return (long if positive, short if negative)
    results['Strategy_Return'] = results['Signal'] * results['Actual_Return']
    results['Benchmark_Return'] = results['Actual_Return'] # Buy & Hold benchmark

    # Calculate cumulative returns
    results['Cumulative_Strategy_Return'] = (1 + results['Strategy_Return']).cumprod()
    results['Cumulative_Benchmark_Return'] = (1 + results['Benchmark_Return']).cumprod()

    # Diagnostic prints for auditing returns (first 10 rows)
    if print_results:
        print("\n--- Diagnostic: Sample Returns (First 10) ---")
        print(results[['Actual_Return', 'Predicted_Return', 'Strategy_Return', 'Cumulative_Strategy_Return']].head(10))
        print(f"Final Cumulative Strategy Return (multiplier): {results['Cumulative_Strategy_Return'].iloc[-1]:.4f}")
        print(f"Final Cumulative Benchmark Return (multiplier): {results['Cumulative_Benchmark_Return'].iloc[-1]:.4f}")

    # Calculate final returns (subtract 1 to get percentage return from multiplier)
    final_strategy_return = results['Cumulative_Strategy_Return'].iloc[-1] - 1
    final_benchmark_return = results['Cumulative_Benchmark_Return'].iloc[-1] - 1

    days = len(results)
    # Annualize returns assuming 252 trading days in a year
    annualized_strategy_return = (1 + final_strategy_return)**(252/days) - 1
    annualized_benchmark_return = (1 + final_benchmark_return)**(252/days) - 1

    # Annualize volatility (standard deviation of daily returns)
    annualized_strategy_vol = results['Strategy_Return'].std() * np.sqrt(252)
    annualized_benchmark_vol = results['Benchmark_Return'].std() * np.sqrt(252)

    # Calculate Sharpe Ratio (Risk-adjusted return). Assume risk-free rate is 0 for simplicity.
    sharpe_strategy = annualized_strategy_return / annualized_strategy_vol if annualized_strategy_vol != 0 else 0
    sharpe_benchmark = annualized_benchmark_return / annualized_benchmark_vol if annualized_benchmark_vol != 0 else 0

    # Calculate directional accuracy: how often the predicted direction matches the actual direction
    correct_direction = np.sum(np.sign(results['Predicted_Return']) == np.sign(results['Actual_Return']))
    direction_accuracy = correct_direction / len(results) * 100

    metrics = {
        'Final Cumulative Return (Strategy)': final_strategy_return,
        'Final Cumulative Return (Benchmark)': final_benchmark_return,
        'Annualized Return (Strategy)': annualized_strategy_return,
        'Annualized Return (Benchmark)': annualized_benchmark_return,
        'Annualized Volatility (Strategy)': annualized_strategy_vol,
        'Annualized Volatility (Benchmark)': annualized_benchmark_vol,
        'Sharpe Ratio (Strategy)': sharpe_strategy,
        'Sharpe Ratio (Benchmark)': sharpe_benchmark,
        'Directional Accuracy (%)': direction_accuracy
    }

    if print_results:
        metrics_df = pd.DataFrame({
            'Metric': ['Final Cumulative Return', 'Annualized Return', 'Annualized Volatility', 'Sharpe Ratio', 'Directional Accuracy (%)'],
            'Model Strategy': [f"{metrics['Final Cumulative Return (Strategy)']:.2%}", f"{metrics['Annualized Return (Strategy)']:.2%}", f"{metrics['Annualized Volatility (Strategy)']:.2%}", f"{metrics['Sharpe Ratio (Strategy)']:.2f}", f"{metrics['Directional Accuracy (%)']:.2f}"],
            'Buy & Hold Benchmark': [f"{metrics['Final Cumulative Return (Benchmark)']:.2%}", f"{metrics['Annualized Return (Benchmark)']:.2%}", f"{metrics['Annualized Volatility (Benchmark)']:.2%}", f"{metrics['Sharpe Ratio (Benchmark)']:.2f}", "-"]
        })
        print(metrics_df.to_string(index=False))

        # Plot cumulative returns
        plt.figure(figsize=(14, 7))
        plt.plot(results.index, results['Cumulative_Strategy_Return'], label='Model Strategy', color='green')
        plt.plot(results.index, results['Cumulative_Benchmark_Return'], label='Buy & Hold Benchmark', color='blue')
        plt.title(f'Trading Strategy Performance: Model vs. Benchmark for {ticker}')
        plt.xlabel('Test Sample Index') # Using index as x-axis for simplicity, can be dates if preferred
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    return metrics


# --- Main Pipeline Execution ---

def run_pipeline(ticker='AAPL', period='2y', seq_length=40, wavelet='db3', level=2,
                 epochs=50, batch_size=32, lr=0.001, print_backtest=True):
    """
    Main pipeline to run the stock prediction and backtesting process for a single ticker.
    Ensures a leak-proof data handling by splitting raw data first, then computing features.

    Args:
        ticker (str): The stock ticker symbol.
        period (str): The historical period for data fetching.
        seq_length (int): The length of input sequences for the Transformer.
        wavelet (str): The wavelet to use for WPD.
        level (int): The decomposition level for WPD.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for DataLoader.
        lr (float): Learning rate for the optimizer.
        print_backtest (bool): Whether to print detailed backtest results and plots.

    Returns:
        dict: Performance metrics for the given ticker.
    """
    print(f"\n--- Starting pipeline for {ticker} ---")

    # 1. Prepare Raw Combined Data (Stock + Macro)
    try:
        raw_combined = create_raw_combined_dataset(ticker, period=period)
    except ValueError as e:
        print(f"Skipping {ticker} due to data fetching/preparation error: {e}")
        return None

    # 2. Chronological Split on Raw Data to prevent lookahead bias
    dataset_size = len(raw_combined)
    if dataset_size < seq_length * 2: # Ensure enough data for sequences and splits
        print(f"Skipping {ticker}: Not enough data ({dataset_size} samples) for sequence length {seq_length}.")
        return None

    train_size = int(0.7 * dataset_size)
    val_size = int(0.15 * dataset_size)
    # Ensure test set has at least seq_length + 1 for target
    if train_size + val_size + seq_length + 1 > dataset_size:
        print(f"Adjusting split sizes for {ticker} due to insufficient data for test set.")
        # Re-calculate to ensure minimum test set size
        test_size = max(seq_length + 1, dataset_size - train_size - val_size)
        val_size = int(0.15 * (dataset_size - test_size))
        train_size = dataset_size - val_size - test_size
        if train_size <= 0 or val_size <= 0 or test_size <= 0:
             print(f"Skipping {ticker}: Cannot create valid splits with current data and sequence length.")
             return None

    raw_train = raw_combined.iloc[:train_size]
    raw_val = raw_combined.iloc[train_size : train_size + val_size]
    raw_test = raw_combined.iloc[train_size + val_size : train_size + val_size + test_size]

    # Ensure splits are not empty
    if raw_train.empty or raw_val.empty or raw_test.empty:
        print(f"Skipping {ticker}: One or more data splits are empty after initial division.")
        return None

    print(f"Data split sizes: Train={len(raw_train)}, Val={len(raw_val)}, Test={len(raw_test)}")

    # 3. Compute Features and Target Per Split (crucial for preventing data leakage)
    print("Computing features per split...")
    train_df = compute_features_per_split(raw_train)
    val_df = compute_features_per_split(raw_val)
    test_df = compute_features_per_split(raw_test)

    # Ensure feature-engineered dataframes are not empty after dropping NaNs
    if train_df.empty or val_df.empty or test_df.empty:
        print(f"Skipping {ticker}: One or more feature-engineered dataframes are empty. This can happen if the initial raw data is too short or has many NaNs.")
        return None

    # 4. Apply WPD Separately to Each Split (prevents signal leakage across splits)
    print(f"Applying WPD and preparing sequences for {ticker}...")
    train_features, train_target = apply_wpd_to_data(train_df, wavelet=wavelet, level=level)
    val_features, val_target = apply_wpd_to_data(val_df, wavelet=wavelet, level=level)
    test_features, test_target = apply_wpd_to_data(test_df, wavelet=wavelet, level=level)

    # Prepare sequences for the Transformer model
    X_train, y_train = prepare_sequences(train_features, train_target, seq_length=seq_length)
    X_val, y_val = prepare_sequences(val_features, val_target, seq_length=seq_length)
    X_test, y_test = prepare_sequences(test_features, test_target, seq_length=seq_length)

    # Check if sequences are empty after preparation
    if X_train.size == 0 or X_val.size == 0 or X_test.size == 0:
        print(f"Skipping {ticker}: Not enough data to create sequences after WPD and sequence preparation. (Min {seq_length} data points needed per sequence + target)")
        return None

    # 5. Correct Scaling: Fit ONLY on training data, then transform all sets
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    # Reshape X data for scaler (MinMaxScaler expects 2D array)
    ns_train, sl_train, nf_train = X_train.shape
    X_train_reshaped = X_train.reshape(ns_train, sl_train * nf_train)
    X_train_scaled = scaler_X.fit_transform(X_train_reshaped).reshape(ns_train, sl_train, nf_train)

    ns_val, sl_val, nf_val = X_val.shape
    X_val_reshaped = X_val.reshape(ns_val, sl_val * nf_val)
    X_val_scaled = scaler_X.transform(X_val_reshaped).reshape(ns_val, sl_val, nf_val)

    ns_test, sl_test, nf_test = X_test.shape
    X_test_reshaped = X_test.reshape(ns_test, sl_test * nf_test)
    X_test_scaled = scaler_X.transform(X_test_reshaped).reshape(ns_test, sl_test, nf_test)

    # Scale target variables
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_val_scaled = scaler_y.transform(y_val)
    y_test_scaled = scaler_y.transform(y_test)

    # 6. Create PyTorch Datasets and DataLoaders
    train_dataset = WPDDataset(X_train_scaled, y_train_scaled)
    val_dataset = WPDDataset(X_val_scaled, y_val_scaled)
    test_dataset = WPDDataset(X_test_scaled, y_test_scaled)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 7. Initialize and Train Model
    input_dim = X_train.shape[2] # Number of features per time step
    model = WPDTransformer(input_dim, seq_length).to(device)
    model = train_model(model, train_loader, val_loader, epochs=epochs, lr=lr)

    # 8. Evaluate Model and Perform Backtesting
    predictions, actuals = evaluate_model(model, test_loader, scaler_y)

    # Calculate standard regression metrics
    mse = mean_squared_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)
    print(f"\n--- Model Prediction Quality for {ticker} ---")
    print(f"Test MSE: {mse:.6f}, R² Score: {r2:.4f}")

    # Perform backtesting and get financial metrics
    metrics = backtest_strategy(predictions, actuals, ticker=ticker, print_results=print_backtest)

    return metrics


# --- Automated Testing on a Basket of Stocks ---

def automate_testing(period='2y', seq_length=40, epochs=50, batch_size=32, lr=0.001):
    """
    Automates the entire stock prediction and backtesting process for a predefined basket of stocks.
    Collects and aggregates performance metrics, providing overall and sector-wise summaries.

    Args:
        period (str): Historical period for data fetching.
        seq_length (int): Sequence length for the Transformer.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for DataLoader.
        lr (float): Learning rate.
    """
    # Define sectors and a list of stocks for each sector
    # Customize this dictionary with your desired stocks and sectors
    sector_stocks = {
        'Technology': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'AMD', 'INTC', 'CSCO', 'ADBE', 'ORCL', 'IBM'],
        'Healthcare': ['JNJ', 'PFE', 'MRK', 'ABT', 'LLY', 'UNH', 'TMO', 'MDT', 'BMY', 'GILD'],
        'Financials': ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'AXP', 'BLK', 'SCHW', 'PNC'],
        'Consumer Discretionary': ['AMZN', 'TSLA', 'HD', 'NKE', 'SBUX', 'MCD', 'LOW', 'TJX', 'F', 'GM'],
        'Industrials': ['BA', 'CAT', 'GE', 'HON', 'MMM', 'UNP', 'LMT', 'FDX', 'DE', 'CSX'],
        'Energy': ['XOM', 'CVX', 'SLB', 'COP', 'EOG', 'OXY', 'VLO', 'MPC', 'PSX', 'KMI'],
        'Utilities': ['NEE', 'DUK', 'SO', 'D', 'EXC', 'AEP', 'SRE', 'XEL', 'PEG', 'ED'],
        'Consumer Staples': ['PG', 'KO', 'PEP', 'WMT', 'COST', 'PM', 'MO', 'CL', 'KMB', 'GIS'],
        'Communication Services': ['GOOG', 'META', 'NFLX', 'DIS', 'VZ', 'T', 'CMCSA', 'CHTR', 'EA', 'ATVI'],
        'Materials': ['LIN', 'SHW', 'ECL', 'DOW', 'NEM', 'FCX', 'APD', 'NUE', 'MLM', 'VMC']
    }

    all_results = []

    # Flatten the list of stocks for progress tracking with tqdm
    flat_stocks = [(sector, ticker) for sector, tickers in sector_stocks.items() for ticker in tickers]

    # Iterate through each stock and run the pipeline
    for sector, ticker in tqdm(flat_stocks, desc="Testing Stocks"):
        try:
            # Run the pipeline, suppressing individual backtest plots for bulk testing
            metrics = run_pipeline(ticker=ticker, period=period, seq_length=seq_length, epochs=epochs,
                                   batch_size=batch_size, lr=lr, print_backtest=False)
            if metrics: # Only append if pipeline ran successfully and returned metrics
                metrics['Ticker'] = ticker
                metrics['Sector'] = sector
                all_results.append(metrics)
        except Exception as e:
            print(f"Error processing {ticker} in {sector}: {e}")
            # Continue to the next stock even if one fails

    # Create a DataFrame from collected results for easy analysis
    results_df = pd.DataFrame(all_results)

    if results_df.empty:
        print("\nNo results collected. Please check for errors during pipeline execution.")
        return

    # --- Overall Summary ---
    print("\n" + "="*30)
    print("--- Overall Summary ---")
    print("="*30)
    overall_avg = results_df.mean(numeric_only=True)
    print("Average Metrics Across All Successfully Processed Stocks:")
    for key, value in overall_avg.items():
        if 'Return' in key or 'Volatility' in key or 'Accuracy' in key:
            print(f"{key}: {value:.2%}")
        else:
            print(f"{key}: {value:.2f}")

    # --- Per-Sector Summary ---
    print("\n" + "="*30)
    print("--- Per-Sector Summary ---")
    print("="*30)
    # Group results by sector and calculate mean metrics for each sector
    sector_summary = results_df.groupby('Sector').mean(numeric_only=True)
    print(sector_summary.to_string())

    # --- Analysis: Top/Underperforming Stocks by Sharpe Ratio ---
    print("\n" + "="*30)
    print("--- Top 5 Stocks by Sharpe Ratio (Strategy) ---")
    print("="*30)
    top_sharpe = results_df.sort_values('Sharpe Ratio (Strategy)', ascending=False).head(5)
    print(top_sharpe[['Ticker', 'Sector', 'Sharpe Ratio (Strategy)', 'Sharpe Ratio (Benchmark)']].to_string(index=False))

    print("\n" + "="*30)
    print("--- Bottom 5 Stocks by Sharpe Ratio (Strategy) ---")
    print("="*30)
    bottom_sharpe = results_df.sort_values('Sharpe Ratio (Strategy)', ascending=True).head(5)
    print(bottom_sharpe[['Ticker', 'Sector', 'Sharpe Ratio (Strategy)', 'Sharpe Ratio (Benchmark)']].to_string(index=False))

    # --- Visualization: Average Sharpe Ratio per Sector ---
    plt.figure(figsize=(12, 6))
    # Plot strategy Sharpe Ratio
    sector_summary['Sharpe Ratio (Strategy)'].plot(kind='bar', color='green', label='Strategy')
    # Plot benchmark Sharpe Ratio with transparency
    sector_summary['Sharpe Ratio (Benchmark)'].plot(kind='bar', color='blue', label='Benchmark', alpha=0.5)
    plt.title('Average Sharpe Ratio: Model Strategy vs. Buy-and-Hold by Sector')
    plt.ylabel('Sharpe Ratio')
    plt.xlabel('Sector')
    plt.legend()
    plt.xticks(rotation=45, ha='right') # Rotate x-axis labels for readability
    plt.tight_layout() # Adjust layout to prevent labels from overlapping
    plt.show()

    # Save aggregated results to a CSV file for further offline analysis
    results_df.to_csv('stock_basket_results.csv', index=False)
    print("\nAggregated results saved to 'stock_basket_results.csv'")


# --- Script Execution Entry Point ---
if __name__ == '__main__':
    # Example of running the pipeline for a single stock
    # Uncomment the block below and comment out the automate_testing call if you want to test one stock
    # print("\n" + "#"*50)
    # print("### Running single stock pipeline (MSFT) ###")
    # print("#"*50 + "\n")
    # run_pipeline(
    #     ticker='MSFT', # Change the ticker here to any stock symbol
    #     period='2y',
    #     seq_length=40,
    #     epochs=50,
    #     batch_size=32,
    #     lr=0.001,
    #     print_backtest=True # Set to False if you don't want the individual plot
    # )

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
