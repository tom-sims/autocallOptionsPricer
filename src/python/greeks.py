import numpy as np
from monte_carlo import generate_paths_bs, price_by_mc
from autocall import price_autocall

def delta(S0, r, T, sigma, n_steps, n_sims, autocall_params, epsilon=1.0):
    paths_up = generate_paths_bs(S0 + epsilon, T, r, sigma, n_steps, n_sims)
    paths_down = generate_paths_bs(S0 - epsilon, T, r, sigma, n_steps, n_sims)
    
    payoffs_up = price_autocall(paths_up, r, **autocall_params)
    payoffs_down = price_autocall(paths_down, r, **autocall_params)
    
    price_up, _ = price_by_mc(payoffs_up, r, T)
    price_down, _ = price_by_mc(payoffs_down, r, T)
    
    return (price_up - price_down) / (2 * epsilon)

def gamma(S0, r, T, sigma, n_steps, n_sims, autocall_params, epsilon=1.0):
    paths_up = generate_paths_bs(S0 + epsilon, T, r, sigma, n_steps, n_sims)
    paths_mid = generate_paths_bs(S0, T, r, sigma, n_steps, n_sims)
    paths_down = generate_paths_bs(S0 - epsilon, T, r, sigma, n_steps, n_sims)
    
    payoffs_up = price_autocall(paths_up, r, **autocall_params)
    payoffs_mid = price_autocall(paths_mid, r, **autocall_params)
    payoffs_down = price_autocall(paths_down, r, **autocall_params)
    
    price_up, _ = price_by_mc(payoffs_up, r, T)
    price_mid, _ = price_by_mc(payoffs_mid, r, T)
    price_down, _ = price_by_mc(payoffs_down, r, T)
    
    return (price_up - 2 * price_mid + price_down) / epsilon**2

def vega(S0, r, T, sigma, n_steps, n_sims, autocall_params, epsilon=0.01):
    paths_up = generate_paths_bs(S0, T, r, sigma + epsilon, n_steps, n_sims)
    paths_down = generate_paths_bs(S0, T, r, sigma - epsilon, n_steps, n_sims)
    
    payoffs_up = price_autocall(paths_up, r, **autocall_params)
    payoffs_down = price_autocall(paths_down, r, **autocall_params)
    
    price_up, _ = price_by_mc(payoffs_up, r, T)
    price_down, _ = price_by_mc(payoffs_down, r, T)
    
    return (price_up - price_down) / (2 * epsilon)

def theta(S0, r, T, sigma, n_steps, n_sims, autocall_params, epsilon=1/365):
    paths_base = generate_paths_bs(S0, T, r, sigma, n_steps, n_sims)
    paths_new = generate_paths_bs(S0, T - epsilon, r, sigma, n_steps, n_sims)
    
    payoffs_base = price_autocall(paths_base, r, **autocall_params)
    payoffs_new = price_autocall(paths_new, r, **autocall_params)
    
    price_base, _ = price_by_mc(payoffs_base, r, T)
    price_new, _ = price_by_mc(payoffs_new, r, T)
    
    return (price_new - price_base) / epsilon