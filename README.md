# Hawkes Processes for Order Flow Modeling

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Research-orange)](https://github.com/)

##  Overview

This project implements a **Univariate Hawkes Process** to model the arrival times of high-frequency trades (tick data). Unlike Poisson processes, which assume event independence, Hawkes processes capture the **"self-exciting"** nature of financial markets where a large trade increases the probability of subsequent trades.

This tool is designed to estimate the **Branching Ratio ($n$)**, a critical metric in market microstructure that quantifies the degree of reflexivity (endogeneity) in the order book.

**Key Features:**
* **Recursive MLE:** Custom $O(N)$ Log-Likelihood implementation (vs. naive $O(N^2)$).
* **Ogata's Thinning:** Simulation engine for generating synthetic "flash crash" scenarios.
* **Criticality Detection:** Automated calculation of market stability metrics.

##  Mathematical Theory

The conditional intensity function $\lambda(t)$ is defined as:

$$\lambda(t) = \mu + \sum_{t_i < t} \alpha e^{-\beta(t - t_i)}$$

Where:
* $\mu$: Background intensity (exogenous events, e.g., news).
* $\alpha$: Excitation parameter (immediate impact of a trade).
* $\beta$: Decay rate (how fast the market "forgets" the trade).

The **Branching Ratio** ($n$) determines market stability:
$$n = \frac{\alpha}{\beta}$$

* If $n < 1$: Sub-critical (Stable).
* If $n \ge 1$: Super-critical (Unstable / Flash Crash Prone).

## Project Structure
* src/model.py: Core Hawkes logic, including the recursive log-likelihood function.
* src/data_gen.py: Synthetic tick data generator using Ogata's Thinning Algorithm.
* src/analytics.py: Visualization tools for intensity curves and criticality reporting.

## Disclaimer


Educational Use Only. This software is for research and educational purposes. It is not intended to be used as a standalone trading system or financial advice. The "Criticality" metrics derived here are theoretical constructs and may not predict all market anomalies.
