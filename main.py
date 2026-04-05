import matplotlib.pyplot as plt

from config.base_config import SIMULATION_CONFIGURATION
from src.price_process import gbm_price_sim
from src.hedging import sim_delta_hedge_short_call

def main():
    # Unpack simulation configuration
    S0 = SIMULATION_CONFIGURATION['S0']
    mu = SIMULATION_CONFIGURATION['mu']
    sigma = SIMULATION_CONFIGURATION['sigma']
    T = SIMULATION_CONFIGURATION['T']
    n_steps = SIMULATION_CONFIGURATION['n_steps']
    seed = SIMULATION_CONFIGURATION['seed']

    K = S0
    r = 0.03

    # Simulate the GBM price path
    price_path = gbm_price_sim(S0, T, mu, sigma, n_steps, seed)

    results = sim_delta_hedge_short_call(
    price_path=price_path,
    K=K,
    T=T,
    r=r,
    sigma=sigma
)
    
    print("Final portfolio value:", results["final_portfolio_value"])
    print("Final call payoff:", results["final_call_payoff"])

    # Plot the price path
    #plt.figure(figsize=(10, 6))
    #plt.plot(price_path)
    #plt.title('Simulated Geometric Brownian Motion Price Path')
    #plt.xlabel('Time Steps')
    #plt.ylabel('Price')
    #plt.grid()
    #plt.show()

    # test-plotting the portfolio value
    plt.figure(figsize=(10, 6))
    plt.plot(results["times"], results["portfolio_values"])
    plt.title("Portfolio Value (Delta Hedging)")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()