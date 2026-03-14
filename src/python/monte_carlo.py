import numpy as np
from dupire import dupire_local_vol

def generate_paths_bs(S0, T, r, sigma, n_steps, n_sims):
    dt = T / n_steps
    paths = np.zeros((n_sims, n_steps + 1))
    paths[:, 0] = S0
    
    Z = np.random.randn(n_sims, n_steps)
    Z = np.vstack([Z, -Z])
    paths = np.zeros((Z.shape[0], n_steps + 1))
    paths[:, 0] = S0
    
    for t in range(1, n_steps + 1):
        paths[:, t] = paths[:, t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, t-1])
    
    return paths

def generate_paths_dupire(S0, T, r, T_axis, K_axis, iv_surface, n_steps, n_sims):
    dt = T / n_steps
    
    Z = np.random.randn(n_sims, n_steps)
    Z = np.vstack([Z, -Z])
    paths = np.zeros((Z.shape[0], n_steps + 1))
    paths[:, 0] = S0
    
    for t in range(1, n_steps + 1):
        current_T = T - (t - 1) * dt
        for i in range(Z.shape[0]):
            S = paths[i, t-1]
            local_vol = dupire_local_vol(T_axis, K_axis, iv_surface, current_T, S)
            paths[i, t] = S * np.exp((r - 0.5 * local_vol**2) * dt + local_vol * np.sqrt(dt) * Z[i, t-1])
    
    return paths

def price_by_mc(payoffs, r, T):
    price = np.exp(-r * T) * np.mean(payoffs)
    std_err = np.std(payoffs) / np.sqrt(len(payoffs))
    return price, std_err