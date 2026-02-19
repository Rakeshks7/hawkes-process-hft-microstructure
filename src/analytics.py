import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_intensity(model, timestamps, t_start=0, t_end=100):
    t_grid = np.linspace(t_start, t_end, 1000)
    lambda_grid = []

    relevant_history = timestamps[timestamps < t_end]
    
    mu = model.params.mu
    alpha = model.params.alpha
    beta = model.params.beta
    
    for t in t_grid:
        history_mask = relevant_history < t
        if not np.any(history_mask):
            lambda_grid.append(mu)
        else:
            decay = np.sum(np.exp(-beta * (t - relevant_history[history_mask])))
            lambda_grid.append(mu + alpha * decay)

    plt.figure(figsize=(12, 6))

    plt.plot(t_grid, lambda_grid, label='$\lambda(t)$ Intensity', color='#2c3e50', lw=2)

    events_in_window = timestamps[(timestamps >= t_start) & (timestamps <= t_end)]
    plt.scatter(events_in_window, [min(lambda_grid)] * len(events_in_window), 
                marker='|', color='#e74c3c', s=100, label='Trade Events', alpha=0.6)
    
    plt.title("Hawkes Process Intensity: Modeling Order Flow Clustering", fontsize=14)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Conditional Intensity (Probability of Trade)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def report_criticality(branching_ratio):
    print("\n" + "="*40)
    print(f"MARKET CRITICALITY REPORT")
    print("="*40)
    print(f"Branching Ratio (n): {branching_ratio:.4f}")
    
    if branching_ratio >= 1.0:
        print("STATUS: [CRITICAL / UNSTABLE]")
        print("Interpretation: The market is endogenously unstable.")
        print("Implication: High probability of a self-fueled flash crash.")
    elif branching_ratio > 0.7:
        print("STATUS: [ELEVATED]")
        print("Interpretation: High degree of reflexivity.")
        print("Implication: Significant volatility clustering expected.")
    else:
        print("STATUS: [STABLE]")
        print("Interpretation: Market is driven mostly by external news (Poisson), not internal feedback.")