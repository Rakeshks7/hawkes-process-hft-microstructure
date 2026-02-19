import numpy as np
from scipy.optimize import minimize
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class HawkesParams:
    mu: float      
    alpha: float   
    beta: float    

class UnivariateHawkes:
    def __init__(self):
        self.params = None
        self.timestamps = None

    def _recursive_log_likelihood(self, params, timestamps):
        mu, alpha, beta = params

        if mu <= 0 or alpha < 0 or beta <= 0:
            return 1e9

        n = len(timestamps)
        T = timestamps[-1]

        term1 = mu * T
        term2 = (alpha / beta) * np.sum(1 - np.exp(-beta * (T - timestamps)))
        integral_term = term1 + term2
        
        log_lambda_sum = 0.0
        r_prev = 0.0 # Recursive term

        log_lambda_sum += np.log(mu)
        
        for i in range(1, n):
            dt = timestamps[i] - timestamps[i-1]
            r_curr = np.exp(-beta * dt) * (r_prev + alpha)
            
            lam = mu + r_curr
            if lam <= 0: return 1e9 # Safety
            
            log_lambda_sum += np.log(lam)
            r_prev = r_curr

        return -(log_lambda_sum - integral_term)

    def fit(self, timestamps: np.ndarray) -> HawkesParams:
        self.timestamps = np.sort(timestamps)
        T = self.timestamps[-1]

        initial_guess = [len(timestamps) / T * 0.5, 0.1, 1.0]

        bounds = ((1e-5, None), (1e-5, None), (1e-5, None))
        
        logger.info("Starting MLE optimization...")
        result = minimize(
            self._recursive_log_likelihood,
            initial_guess,
            args=(self.timestamps,),
            bounds=bounds,
            method='L-BFGS-B'
        )
        
        if not result.success:
            logger.warning(f"Optimization failed: {result.message}")
        
        self.params = HawkesParams(*result.x)
        logger.info(f"Fit Complete: {self.params}")
        return self.params

    def get_branching_ratio(self) -> float:
        if not self.params:
            raise ValueError("Model not fitted yet.")
        return self.params.alpha / self.params.beta

    def simulate(self, mu, alpha, beta, T):
        timestamps = []
        t = 0

        history = []

        while t < T:
            current_lambda = mu + sum(alpha * np.exp(-beta * (t - ti)) for ti in history)

            w = -np.log(np.random.uniform()) / current_lambda
            t += w
            
            if t > T: break

            exact_lambda = mu + sum(alpha * np.exp(-beta * (t - ti)) for ti in history)
            
            if np.random.uniform() < exact_lambda / current_lambda:
                timestamps.append(t)
                history.append(t)
                
        return np.array(timestamps)