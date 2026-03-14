import numpy as np
import pandas as pd
from scipy.interpolate import RectBivariateSpline
from scipy.stats import norm
from black_scholes import bs_implied_vol, bs_price

def load_options_chain(filepath):
    df = pd.read_csv(filepath)
    
    calls = df[["Expiration Date", "Strike", "Bid", "Ask", "Volume", "IV", "Open Interest"]].copy()
    calls["option_type"] = "call"
    
    puts = df[["Expiration Date", "Strike", "Bid.1", "Ask.1", "Volume.1", "IV.1", "Open Interest.1"]].copy()
    puts["option_type"] = "put"

    puts.rename(columns={
        "Bid.1": "Bid",
        "Ask.1": "Ask",
        "Volume.1": "Volume",
        "IV.1": "IV",
        "Open Interest.1": "Open Interest"
    }, inplace=True)

    combined = pd.concat([calls, puts], ignore_index=True)
    combined.columns = combined.columns.str.lower().str.replace(" ", "_")
    
    return combined

def filter_options(df, spot, min_volume=10, min_open_interest=100):
    df = df.copy()
    
    df["expiration_date"] = pd.to_datetime(df["expiration_date"])
    today = pd.Timestamp.today().normalize()
    df["T"] = (df["expiration_date"] - today).dt.days / 365
    
    df = df[df["T"] > 7/365]
    df = df[df["volume"] >= min_volume]
    df = df[df["open_interest"] >= min_open_interest]
    df["mid"] = (df["bid"] + df["ask"]) / 2
    df = df[df["mid"] > 0]
    df = df[df["strike"] >= spot * 0.8]
    df = df[df["strike"] <= spot * 1.2]
    
    return df.reset_index(drop=True)

def compute_implied_vols(df, spot, r=0.05):
    df = df.copy()
    
    def compute_iv(row):
        try:
            return bs_implied_vol(
                market_price=row["mid"],
                S=spot,
                K=row["strike"],
                T=row["T"],
                r=r,
                option_type=row["option_type"]
            )
        except Exception:
            return np.nan
    
    df["iv_computed"] = df.apply(compute_iv, axis=1)
    df = df.dropna(subset=["iv_computed"])
    df = df[df["iv_computed"] > 0]
    
    return df.reset_index(drop=True)

def build_vol_surface(df):
    calls = df[df["option_type"] == "call"].copy()
    
    T_axis = np.array(sorted(calls["T"].unique()))
    K_axis = np.array(sorted(calls["strike"].unique()))
    
    iv_surface = np.full((len(T_axis), len(K_axis)), np.nan)
    
    for i, T in enumerate(T_axis):
        for j, K in enumerate(K_axis):
            match = calls[(calls["T"] == T) & (calls["strike"] == K)]
            if not match.empty:
                iv_surface[i, j] = match["iv_computed"].values[0]
    
    valid_T = ~np.all(np.isnan(iv_surface), axis=1)
    valid_K = ~np.all(np.isnan(iv_surface), axis=0)
    iv_surface = iv_surface[np.ix_(valid_T, valid_K)]
    T_axis = T_axis[valid_T]
    K_axis = K_axis[valid_K]
    
    col_means = np.nanmean(iv_surface, axis=0)
    nan_mask = np.isnan(iv_surface)
    iv_surface[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
    
    return T_axis, K_axis, iv_surface

def interpolate_vol_surface(T_axis, K_axis, iv_surface, T_query, K_query):
    spline = RectBivariateSpline(T_axis, K_axis, iv_surface)
    iv = float(spline(T_query, K_query))
    return max(iv, 0.01)

def dupire_local_vol(T_axis, K_axis, iv_surface, T, K, r=0.05):
    dT = 1e-4
    dK = 1.0

    def call_price(t, k):
        iv = interpolate_vol_surface(T_axis, K_axis, iv_surface, t, k)
        return bs_price(S=K, K=k, T=t, r=r, sigma=iv, option_type="call")

    dC_dT = (call_price(T + dT, K) - call_price(T - dT, K)) / (2 * dT)
    d2C_dK2 = (call_price(T, K + dK) - 2 * call_price(T, K) + call_price(T, K - dK)) / (dK ** 2)

    denominator = 0.5 * K ** 2 * d2C_dK2

    if denominator <= 0:
        return 0.01

    local_var = dC_dT / denominator
    
    if local_var <= 0:
        return 0.01

    return np.sqrt(local_var)

if __name__ == "__main__":
    spot = 6632.19
    df = load_options_chain("data/spx_quotedata.csv")
    df = filter_options(df, spot)
    df = compute_implied_vols(df, spot)
    T_axis, K_axis, iv_surface = build_vol_surface(df)
    
    print(f"Rows after filtering: {len(df)}")
    print(f"T_axis: {T_axis}")
    print(f"K_axis: {K_axis}")
    print(f"Surface shape: {iv_surface.shape}")
    
    test_T = T_axis[len(T_axis) // 2]
    test_K = K_axis[len(K_axis) // 2]
    iv = interpolate_vol_surface(T_axis, K_axis, iv_surface, test_T, test_K)
    lv = dupire_local_vol(T_axis, K_axis, iv_surface, test_T, test_K)
    print(f"Implied vol at T={test_T:.2f}, K={test_K:.0f}: {iv:.4f}")
    print(f"Local vol at T={test_T:.2f}, K={test_K:.0f}: {lv:.4f}")