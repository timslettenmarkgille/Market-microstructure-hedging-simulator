import matplotlib.pyplot as plt

from config.base_config import SIMULATION_CONFIGURATION
from src.price_process import gbm_price_sim

def main():
    # Unpack simulation configuration
    S0 = SIMULATION_CONFIGURATION['S0']
    mu = SIMULATION_CONFIGURATION['mu']
    sigma = SIMULATION_CONFIGURATION['sigma']
    T = SIMULATION_CONFIGURATION['T']
    n_steps = SIMULATION_CONFIGURATION['n_steps']
    seed = SIMULATION_CONFIGURATION['seed']

    # Simulate the GBM price path
    price_path = gbm_price_sim(S0, T, mu, sigma, n_steps, seed)

    # Plot the price path
    plt.figure(figsize=(10, 6))
    plt.plot(price_path)
    plt.title('Simulated Geometric Brownian Motion Price Path')
    plt.xlabel('Time Steps')
    plt.ylabel('Price')
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()