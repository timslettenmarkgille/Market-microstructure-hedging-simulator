
# Initial hedging setup:
# Short one European call
# Hedge using stock + cash
# Rebalance stock position each time step to match call delta
# Cash account finances stock position purchases/sales 
# No transaction costs in this first example
# Final portfolio value = cash + stock_pos * S_T - call_payoff

import numpy as np
from src.derivatives import (
    BS_european_call_price,
    BS_european_call_delta
)

def sim_delta_hedge_short_call(
        price_path: np.ndarray,
        K: float,
        T: float,
        r: float,
        sigma: float
        ):
    
    price_path = np.asarray(price_path , dtype = np.float64)

    if len(price_path) < 2:
        raise ValueError("price_path must contain at least 2 prices.")

    if np.any(price_path <= 0):
        raise ValueError("All prices in price_path must be > 0.")

    n_steps = len(price_path) - 1
    dt = T / n_steps

    times = np.linspace(0.0, T, len(price_path))

    stock_positions = np.zeros(len(price_path))
    cash_account = np.zeros(len(price_path))
    call_values = np.zeros(len(price_path))
    deltas = np.zeros(len(price_path))
    portfolio_values = np.zeros(len(price_path))

    option_position = -1.0

    # At time 0:
    S0 = price_path[0]

    call_value_0 = BS_european_call_price(S0, K, T, r, sigma)
    delta_0 = BS_european_call_delta(S0, K, T, r, sigma)

    call_values[0] = call_value_0
    deltas[0] = delta_0

    # Sell 1 call
    cash_account[0] = call_value_0

    # Buy stock to hedge
    stock_positions[0] = delta_0
    cash_account[0] = cash_account[0] - stock_positions[0] * S0

    # Portfolio value at time 0:
    portfolio_values[0] = (cash_account[0] + stock_positions[0] * S0 + option_position * call_values[0])


    # Rebalance at each time step:
    for i in range(1, len(price_path)):
        S_t = price_path[i]
        t = times[i]
        tau = max(T - t, 0.0)

        cash_account[i] = cash_account[i - 1] * np.exp(r * dt)

        # Current option value and delta
        call_values[i] = BS_european_call_price(S_t, K, tau, r, sigma)
        deltas[i] = BS_european_call_delta(S_t, K, tau, r, sigma)

        # New stock position and updated cash account
        old_stock_pos = stock_positions[i - 1]
        new_stock_pos = deltas[i]
        stock_positions[i] = new_stock_pos
        trade_size = new_stock_pos - old_stock_pos
        cash_account[i] = cash_account[i] - trade_size * S_t

        # Portfolio value at time t:
        portfolio_values[i] = (cash_account[i] + stock_positions[i] * S_t + option_position * call_values[i])

    # Final payoff at maturity:
    S_T = price_path[-1]
    call_payoff = max(S_T - K, 0.0)

    final_portfolio_value = (cash_account[-1] + stock_positions[-1] * S_T - call_payoff)

    return {
        "times": times,
        "price_path": price_path,
        "stock_positions": stock_positions,
        "cash_account": cash_account,
        "call_values": call_values,
        "deltas": deltas,
        "portfolio_values": portfolio_values,
        "final_call_payoff": call_payoff,
        "final_portfolio_value": final_portfolio_value
    }