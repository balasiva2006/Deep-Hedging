import argparse
import os
import numpy as np
import tensorflow as tf
from src.model import LSTMHedger
from src.data_handler import get_stock_params
from src.quant_lib import black_scholes
from src.trainer import run_mse_bootstrapping, run_cvar_tuning

def generate_paths(S0, sigma, r, steps=375, num_paths=100000):
    # Annualized T for one day
    T = (1/365.25) * (steps/375)
    dt = T/steps
    Z = np.random.normal(0, 1, (num_paths, steps))
    drift = (r - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * Z
    S_paths = S0 * np.exp(np.cumsum(drift + diffusion, axis=1))
    S_paths = np.hstack([np.full((num_paths, 1), S0), S_paths])
    
    # Calculate BS Deltas for the paths
    T_array = T - (np.arange(steps) * dt)
    BS_deltas = np.zeros((num_paths, steps))
    for t in range(steps):
        _, d, _, _ = black_scholes(S_paths[:, t], S0, T_array[t], r, sigma, 'call')
        BS_deltas[:, t] = d
    return S_paths, BS_deltas

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=['scratch', 'tune', 'predict'], required=True)
    parser.add_argument("--stock", type=str, required=True)
    parser.add_argument("--csv", type=str)
    parser.add_argument("--weights", type=str)
    args = parser.parse_args()

    os.makedirs(f"checkpoints/{args.stock}", exist_ok=True)
    model = LSTMHedger()
    model(tf.zeros((1, 3)), model.get_initial_states(1)) # Build graph

    if args.mode in ['scratch', 'tune']:
        S0, sigma = get_stock_params(args.csv, args.stock)
        print(f"Processing {args.stock}: S0={S0}, Sigma={sigma:.4f}")
        S_paths, BS_deltas = generate_paths(S0, sigma, r=0.05)

        if args.mode == 'scratch':
            print("Mode: Scratch. Running MSE then CVaR...")
            run_mse_bootstrapping(model, S_paths, BS_deltas, S0)
            run_cvar_tuning(model, S_paths, BS_deltas, S0, r=0.05)
        else:
            print(f"Mode: Tune. Loading {args.weights} and fine-tuning CVaR...")
            model.load_weights(args.weights)
            run_cvar_tuning(model, S_paths, BS_deltas, S0, r=0.05, epochs=5)
        
        stock_folder = f"checkpoints/{args.stock}"
        os.makedirs(stock_folder, exist_ok=True)
        save_path = f"checkpoints/{args.stock}/model_latest.h5"
        model.save_weights(save_path)
        print(f"Weights saved to {save_path}")

    elif args.mode == 'predict':
        model.load_weights(args.weights)
        s_price = float(input("Enter Current Stock Price: "))
        bs_delta = float(input("Enter BS Delta: "))
        k_strike = float(input("Enter Option Strike: "))
        
        # Predict
        inp = tf.constant([[s_price/k_strike, bs_delta, 0.0]], dtype=tf.float32)
        delta, _ = model(inp, model.get_initial_states(1))
        print(f"\n[RESULT] Recommended Hedge Delta: {delta.numpy()[0][  0]:.4f}")

if __name__ == "__main__":
    main()