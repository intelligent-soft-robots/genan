import numpy as np
from scipy.stats import norm


def calculate_success_rate_cis(successes: int, trials: int, alpha=0.05):
    z = norm.ppf(1.0 - alpha / 2)

    p_hat = successes / trials
    den = 1 + z**2 / trials
    center = (p_hat + z**2 / (2 * trials)) / den
    half_width = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * trials)) / trials) / den

    ci_low = center - half_width
    ci_high = center + half_width
    if ci_low < 1e-8:
        ci_low = 0.0
    return ci_low, ci_high
