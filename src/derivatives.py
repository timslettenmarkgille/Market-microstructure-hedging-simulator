import numpy as np
import math


# Black-Scholes formula for European call option price
def BS_european_call_price(
        S: float, 
        K: float, 
        T: float, 
        r: float, 
        sigma: float
        ):
    
    if S <= 0:
        raise ValueError("S must be > 0.")
    if K <= 0:
        raise ValueError("K must be > 0.")
    if sigma < 0:
        raise ValueError("sigma must be >= 0.")
    if T < 0:
        raise ValueError("T must be >= 0.")

    
    if T == 0:
        return max(S - K, 0.0)
    
    if sigma == 0:
        S_T = S * np.exp(r * T)
        payoff = max(S_T - K, 0.0)
        return float(np.exp(-r * T) * payoff)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    call_price = 0.5 * S *(1 + math.erf(d1 / np.sqrt(2))) - np.exp(-r * T) * K * 0.5 * (1 + math.erf(d2 / np.sqrt(2)))
    return call_price


# Black-Scholes formula for European put option price
def BS_european_put_price(
        S: float, 
        K: float, 
        T: float, 
        r: float, 
        sigma: float
        ):
    
    if S <= 0:
        raise ValueError("S must be > 0.")
    if K <= 0:
        raise ValueError("K must be > 0.")
    if sigma < 0:
        raise ValueError("sigma must be >= 0.")
    if T < 0:
        raise ValueError("T must be >= 0.")

    
    if T == 0:
        return max(K - S, 0.0)
    
    if sigma == 0:
        S_T = S * np.exp(r * T)
        payoff = max(K - S_T, 0.0)
        return float(np.exp(-r * T) * payoff)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    put_price = np.exp(-r * T) * K * 0.5 * (1 + math.erf((-d2) / np.sqrt(2))) - 0.5 * S * (1 + math.erf((-d1) / np.sqrt(2)))
    return put_price


# Black-Scholes formula for European call option delta
def BS_european_call_delta(
        S: float, 
        K: float, 
        T: float, 
        r: float, 
        sigma: float
        ):
    
    if S <= 0:
        raise ValueError("S must be > 0.")
    if K <= 0:
        raise ValueError("K must be > 0.")
    if sigma < 0:
        raise ValueError("sigma must be >= 0.")
    if T < 0:
        raise ValueError("T must be >= 0.")

    
    if T == 0:
        if S > K:
            return 1.0
        elif S < K:
            return 0.0
        else:
            return 0.5
    
    if sigma == 0:
        S_T = S * np.exp(r * T)
        if S_T > K:
            return 1.0
        elif S_T < K:
            return 0.0
        else:
            return 0.5

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    
    call_delta = 0.5 * (1 + math.erf(d1 / np.sqrt(2)))
    return call_delta


# Black-Scholes formula for European put option delta
def BS_european_put_delta(
        S: float, 
        K: float, 
        T: float, 
        r: float, 
        sigma: float
        ):
    
    if S <= 0:
        raise ValueError("S must be > 0.")
    if K <= 0:
        raise ValueError("K must be > 0.")
    if sigma < 0:
        raise ValueError("sigma must be >= 0.")
    if T < 0:
        raise ValueError("T must be >= 0.")

    
    if T == 0:
        if S < K:
            return -1.0
        elif S > K:
            return 0.0
        else:
            return -0.5
    
    if sigma == 0:
        S_T = S * np.exp(r * T)
        if S_T < K:
            return -1.0
        elif S_T > K:
            return 0.0
        else:
            return -0.5

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    
    put_delta = 0.5 * (1 + math.erf(d1 / np.sqrt(2))) - 1
    return put_delta

