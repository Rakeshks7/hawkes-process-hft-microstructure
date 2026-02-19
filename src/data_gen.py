import numpy as np
from .model import UnivariateHawkes

def generate_synthetic_order_flow(duration_seconds=3600):
    print(f"Generating {duration_seconds} seconds of synthetic HFT data...")

    true_mu = 0.5    
    true_alpha = 0.8 
    true_beta = 1.2  
    
    simulator = UnivariateHawkes()
    timestamps = simulator.simulate(true_mu, true_alpha, true_beta, duration_seconds)
    
    print(f"Generated {len(timestamps)} trades.")
    return timestamps