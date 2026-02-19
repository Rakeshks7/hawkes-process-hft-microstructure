import numpy as np
from src.data_gen import generate_synthetic_order_flow
from src.model import UnivariateHawkes
from src.analytics import plot_intensity, report_criticality

def main():
    timestamps = generate_synthetic_order_flow(duration_seconds=300)

    hawkes = UnivariateHawkes()

    print("Fitting Hawkes Process to order flow...")
    hawkes.fit(timestamps)

    n = hawkes.get_branching_ratio()

    report_criticality(n)

    print("Generating intensity plot...")
    plot_intensity(hawkes, timestamps, t_start=0, t_end=50)

if __name__ == "__main__":
    main()