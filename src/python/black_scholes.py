import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

def d1(S, K, T, r, sigma):
    return (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

def d2(S, K, T, r, sigma):
    return d1(S, K, T, r, sigma) - (sigma * np.sqrt(T))

def bs_price(S, K, T, r, sigma, option_type):
    d_1 = d1(S, K, T, r, sigma)
    d_2 = d2(S, K, T, r, sigma)
    return (S * norm.cdf(d_1)) - ((K * np.exp(-r * T)) * norm.cdf(d_2)) if option_type == "call" else ((K * np.exp(-r * T)) * norm.cdf(-d_2)) - (S * norm.cdf(-d_1))

def bs_delta(S, K, T, r, sigma, option_type):
    d_1 = d1(S, K, T, r, sigma)
    return norm.cdf(d_1) if option_type == "call" else norm.cdf(d_1) - 1

def bs_gamma(S, K, T, r, sigma):
    d_1 = d1(S, K, T, r, sigma)
    return (norm.pdf(d_1)) / (S * sigma * np.sqrt(T))

def bs_vega(S, K, T, r, sigma):
    d_1 = d1(S, K, T, r, sigma)
    return S * norm.pdf(d_1) * np.sqrt(T)

def bs_theta(S, K, T, r, sigma, option_type):
    d_1 = d1(S, K, T, r, sigma)
    d_2 = d2(S, K, T, r, sigma)
    return ((-S * norm.pdf(d_1) * sigma) / (2 * np.sqrt(T))) - (r * K * np.exp(-r * T)) * (norm.cdf(d_2)) if option_type == "call" else ((-S * norm.pdf(d_1) * sigma) / (2 * np.sqrt(T))) + (r * K * np.exp(-r * T)) * (norm.cdf(-d_2))

def bs_implied_vol(market_price, S, K, T, r, option_type):
    def objective(sigma):
        return bs_price(S, K, T, r, sigma, option_type) - market_price
    
    try:
        return brentq(objective, 0.001, 5.0)
    except ValueError:
        return np.nan