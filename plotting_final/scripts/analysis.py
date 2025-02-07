""" Some results analysis utilities for supplementary plotting. 

@author: frbourassa
May 2023
"""

import numpy as np

# Utility function to find best grid point given analysis results
def find_best_grid_point(all_res_dict, strat="best"):
    stratkey = "MAP " + strat
    conditions = list(all_res_dict.keys())
    best_cond = conditions[0]
    best_cost = all_res_dict[best_cond]["posterior_probs"][stratkey]
    best_p = all_res_dict[best_cond]["param_estimates"][stratkey]
    for cond in conditions:
        cost = all_res_dict[cond]["posterior_probs"][stratkey]
        if cost > best_cost:
            best_cost = cost
            best_cond = cond
            best_p = all_res_dict[cond]["param_estimates"][stratkey]

    return best_cond, np.asarray(best_p), best_cost


# Autocorrelation analysis. Code from emcee's documentation:
# https://emcee.readthedocs.io/en/stable/tutorials/autocorr/
def next_pow_two(n):
    i = 1
    while i < n:
        i = i << 1
    return i


def autocorr_func_1d(x, norm=True):
    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")
    n = next_pow_two(len(x))
    # Compute the FFT and then (from that) the auto-correlation function
    f = np.fft.fft(x - np.mean(x), n=2 * n)
    acf = np.fft.ifft(f * np.conjugate(f))[: len(x)].real
    acf /= 4 * n

    # Optionally normalize
    if norm and acf[0] != 0.0:
        acf /= acf[0]

    return acf

def autocorr_func_avg(y, norm=True):
    """ y is shaped [n_walkers, n_samples] """
    # Compute the autocorrelation fct for each walker
    f = np.zeros(y.shape[1])
    for yy in y:  # loop over walkers
        f += autocorr_func_1d(yy, norm=False)
    f /= len(y)
    if norm:
        f /= f[0]
    return f


# Automated windowing procedure following Sokal (1989)
def auto_window(taus, c):
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        return np.argmin(m)
    return len(taus) - 1


def autocorr_avg(y, c=5.0):
    """ y is shaped [n_walkers, n_samples] """
    # First compute the integrated autocorrelation time
    f = np.zeros(y.shape[1])
    for yy in y:  # loop over walkers
        f += autocorr_func_1d(yy)
    f /= len(y)
    # Use the automatic windowing described by Sokal, stop the
    # sum to compute tau_int at the auto_window position
    # The sume extends from -time to +time, here use symmetry
    # to rewrite t_int = 2*sum_1^{time} + 1
    taus = 2.0 * np.cumsum(f) - 1.0
    window = auto_window(taus, c)
    # This returns the integrated autocorrelation time,
    # equal to 1/2 + correlation time if the decay is exponential
    return taus[window]

