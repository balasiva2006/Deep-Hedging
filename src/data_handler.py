import pandas as pd
import numpy as np

def parse_symbol(symbol):
    option_type = 'call' if 'CE' in symbol else 'put'
    # Keeping notebook logic: extract digits, take last 5
    digits = ''.join(filter(str.isdigit, symbol.split('CE' if 'CE' in symbol else 'PE')[0]))
    strike = float(digits[-5:]) * 100.0 if len(digits) >= 5 else None
    return strike, option_type

def get_stock_params(csv_path, stock_name):
    df = pd.read_csv(csv_path)
    # Filter by user-provided stock name (NIFTY, APPLE, etc.)
    stock_df = df[df['symbol'].str.contains(stock_name)].copy()
    
    # Get Future price
    fut_df = stock_df[stock_df['symbol'].str.contains('FUT')].sort_values('minute_end')
    S0 = fut_df['last_trade_price'].iloc[0]
    
    # Calculate Sigma from today's path
    returns = np.log(fut_df['last_trade_price'] / fut_df['last_trade_price'].shift(1)).dropna()
    sigma = returns.std() * np.sqrt(252 * 375) # Annualized
    
    return S0, sigma