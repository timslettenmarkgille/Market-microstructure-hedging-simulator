import numpy as np
from typing import Optional

def gbm_price_sim(
        S0: float, 
        T: float, 
        mu: float, 
        sigma: float, 
        n_steps: int,
        seed: Optional[int] = None
):
    if seed is not None:
        np.random.seed(seed)

    dt = T / n_steps

    # Draw random samples from a normal distribution
    Z = np.random.standard_normal(n_steps)

    # Calculate log returns
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z

    # Calculate the price
    price_path = S0 * np.exp(np.cumsum(log_returns))
    
    # Add S0
    price_path = np.insert(price_path, 0, S0)

    return price_path