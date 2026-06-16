import numpy as np
from scipy.stats import norm
import tensorflow as tf

def black_scholes(S, K, T, r, sigma, option_type):
    if T <= 0: # Handle zero or negative time to expiration
        if option_type == 'call':
            return max(0, S - K), 0, 0, 0
        else:
            return max(0, K - S), 0, 0, 0

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        theta = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        theta = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return price, delta, gamma, theta

def implied_volatility(observed_price, S, K, T, r, option_type, tol=1e-5, max_iter=100, verbose=False):
    low_vol = 0.001
    high_vol = 5.0
    for i in range(max_iter):
        mid_vol = (low_vol + high_vol) / 2
        bs_price, _, _, _ = black_scholes(S, K, T, r, mid_vol, option_type)
        if abs(bs_price - observed_price) < tol:
            return mid_vol
        if bs_price < observed_price:
            low_vol = mid_vol
        else:
            high_vol = mid_vol
        if (high_vol - low_vol) < tol:
            return mid_vol
    return mid_vol

@tf.function
def calculate_pnl(S_path_batch, deltas_batch, K, r_tf, option_type_str, transaction_cost=0.0015):
    S_path_batch = tf.cast(S_path_batch, tf.float32)
    deltas_batch = tf.cast(deltas_batch, tf.float32)
    K = tf.cast(K, tf.float32)
    price_changes = S_path_batch[:, 1:] - S_path_batch[:, :-1]
    trading_pnl_per_interval = deltas_batch[:, :-1] * price_changes
    cumulative_trading_pnl = tf.reduce_sum(trading_pnl_per_interval, axis=1)
    initial_fee = transaction_cost * S_path_batch[:, 0] * tf.abs(deltas_batch[:, 0])
    delta_changes = tf.abs(deltas_batch[:, 1:] - deltas_batch[:, :-1])
    subsequent_fees = transaction_cost * S_path_batch[:, 1:] * delta_changes
    total_transaction_costs = initial_fee + tf.reduce_sum(subsequent_fees, axis=1)
    S_T = S_path_batch[:, -1]
    # Simple Call payoff logic from notebook
    option_payoff_at_expiry = tf.maximum(0.0, S_T - K)
    total_pnl = -option_payoff_at_expiry + cumulative_trading_pnl - total_transaction_costs
    return total_pnl

@tf.function
def cvar_loss(pnl_values, alpha=0.05):
    sorted_pnl = tf.sort(pnl_values)
    num_samples = tf.cast(tf.shape(sorted_pnl)[0], tf.float32)
    cvar_index = tf.cast(tf.floor(num_samples * alpha), tf.int32)
    worst_pnl_values = sorted_pnl[:cvar_index + 1]
    cvar_value = -tf.reduce_mean(worst_pnl_values)
    return cvar_value