import numpy as np

def price_autocall(paths, r, observation_dates, autocall_barrier, knockin_barrier, coupon, notional=1.0):
    n_sims, n_steps = paths.shape
    S0 = paths[:, 0]
    dt = 1.0 / n_steps
    payoffs = np.zeros(n_sims)

    for i in range(n_sims):
        redeemed = False

        for obs_idx in observation_dates:
            t = obs_idx * dt
            S_obs = paths[i, obs_idx]

            if S_obs >= autocall_barrier * S0[i]:
                payoffs[i] = notional * (1 + coupon)
                payoffs[i] *= np.exp(-r * t)
                redeemed = True
                break

        if not redeemed:
            T = (n_steps - 1) * dt
            S_final = paths[i, -1]
            knocked_in = np.any(paths[i, :] <= knockin_barrier * S0[i])

            if knocked_in:
                payoffs[i] = notional * (S_final / S0[i])
            else:
                payoffs[i] = notional

            payoffs[i] *= np.exp(-r * T)

    return payoffs