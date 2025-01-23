""" Analysis tools to estimate parameters, confidence intervals, etc.
from estimation algorithms raw results.

@author: frbourassa
May 2022
"""
import numpy as np

def marginal_hist_modes(psamples, nburn):
    map_estimate = np.zeros(psamples.shape[0])
    # Find the mode for each component of the parameter vector.
    zscales = np.std(psamples[:, nburn:], axis=1, keepdims=True)
    zscales[np.isnan(zscales)] = 1.0
    n_pm = psamples.shape[0]
    for p in range(n_pm):
        hist, edges = np.histogram(psamples[p, nburn:], bins="doane")
        maxbin = np.argmax(hist)
        # Middle of most likely bin
        map_estimate[p] = (edges[maxbin] + edges[maxbin+1]) / 2.0

    # Log-posterior prob of closest point. z-scale params
    # so the scale of distances is not twisted.
    norm_samples = psamples / zscales
    norm_map = map_estimate[:, None] / zscales
    closest_sample = np.argmin(
                    np.sum((norm_samples - norm_map)**2, axis=0))

    return map_estimate, closest_sample


# Utility function to find the maximum a posteriori from samples
def find_max_posterior(psamples, logposteriors, strat="hist", burn=0.5):
    """
    There are two options:
        strat == 'hist': take the most sampled value (mode of histogram)
            for each parameter.

        strat == 'best': take the single vector sample with largest
            posterior probability. One can use the likelihood instead of the
            posterior if the prior is uniform, because in that case, the two
            probabilities are proportional.

    The options should agree in principle, because the steady-state
    distribution should be the posterior, by construction.

    In practice, the 'best' strategy works much better, because the most
    sampled values of each parameter separately may not produce together
    an outcome close to the data.

    Args:
        psamples (np.ndarray): sampled parameter values from an MCMC run,
            indexed [parameter, sample].
        logposteriors (np.ndarray): log of the posterior probability
            of each parameter sample in psamples.
        strat (str): either 'hist' or 'best'
        burn (float): fraction of time series to reject at the start
            as a "burn-in" phase. Default: only latter half of points are used.
    Returns:
        map_estimate (np.ndarray): maximum a posteriori estimate of the
            parameters.
        post_prob (float): the log-likelihood of the MAP estimate (for method
            'best') or of the closest sample (for 'hist').
            Will need to be recomputed to find actual value for 'hist' method.
    """
    nburn = int(psamples.shape[1] * burn)
    # Strategy is either "hist", to take the mode of the histogram
    if strat == "hist":
        map_estimate, map_idx = marginal_hist_modes(psamples, nburn)
        post_prob = logposteriors[map_idx]

    # or "best", to return the single sample with the highest posterior prob.
    elif strat == "best":
        where_max = np.argmax(logposteriors)
        map_estimate = psamples[:, where_max]
        post_prob = logposteriors[where_max]

    else:
        raise ValueError("strat should be either 'hist' or 'best'.")
    return map_estimate, post_prob


def find_confidence_interval(psamples, lower=0.05, upper=0.95, burn=0.5):
    """ psamples should flatten such that samples from the same iteration
    are contiguous.
    """
    psamples = psamples.flatten()
    # Find lower and upper quantiles
    burn_step = int(burn*psamples.size)
    ci = np.quantile(psamples[burn_step:], q=[lower, upper])
    return ci


if __name__ == "__main__":
    # Check find posterior strategies with dummy samples
    nsamp = int(1e5)
    psamples = np.random.normal(size=4*nsamp).reshape([4, nsamp])
    scales = np.asarray([[1.0, 10.0, 0.1, 1.0]]).T
    centers = np.asarray([[0.0, 5.0, -0.3, 1.0]]).T
    psamples = psamples * scales + centers
    # Just make log-post be the quadratic distance to centers
    logposteriors = -np.sum((psamples - centers)**2, axis=0)

    res0 = find_max_posterior(psamples, logposteriors, strat="hist", burn=0.2)
    print(res0)
