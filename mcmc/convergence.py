""" Analysis tools to check convergence of model parameter estimation
algorithms, such as MCMC

Borrows code from the emcee documentation examples.

@author: frbourassa
May 2022
"""
import numpy as np

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
    # This returns the integrated autocrrelation time,
    # equal to 1/2 + correlation time if the decay is exponential
    return taus[window]


def check_autocorr_convergence(samples, param_names, cond_nice):
    nsamples = samples.shape[-1]
    # Check autocorrelation time estimation as a function of the number
    # of samples used. Before it's reliable, the estimator scales
    # linearly in log-log scale with the number of samples.
    # Compare to estimation with first 50 % of data to check that.
    taus_auto = []
    taus_mid = []
    poor_estimate = False
    print("* CHECKING AUTOCORRELATION TIMES *")
    for i in range(len(param_names)):
        taus_mid.append(autocorr_avg(samples[i, :, :nsamples//2], c=5.0))
        taus_auto.append(autocorr_avg(samples[i], c=5.0))
        error_tau = 1.0 - taus_mid[i]/taus_auto[i]
        # Check discrepancy: if more than 10 % we clearly haven't converged
        # (in the scaling regime, x2 nb pts means x2 correl. time: 50 % error)
        if error_tau > 0.1:
            print("Autocorrelation time estimator has not converged "
                    + "for parameter {}".format(param_names[i]))
            poor_estimate = True
    taus_auto_frac = [nsamples / a for a in taus_auto]

    print("Autocorrelation time for each parameter: {}".format(taus_auto))
    print("A chain duration is thus {} times taus".format(taus_auto_frac))
    if nsamples < 10*max(taus_auto) or poor_estimate:
        #raise ValueError("The chains most likely have not converged yet.")
        print("The chains most likely have not converged yet.")
    else:
        print("* THE CHAINS' CONVERGENCE SEEMS ACCEPTABLE *")
        print("Make sure to discard the first {} % of the chains".format(20.0 / min(taus_auto_frac) * 100.0))
        print()
    return taus_auto


def check_acceptance_fractions(frac, cond, lo=0.05, hi=0.95):
    exit_code = -1
    if np.amin(frac) < lo:
        print("Low acceptance rate ({}) for condition {}".format(np.amin(frac),cond))
    elif np.amax(frac) > hi:
        print("High acceptance rate ({}) for condition {}".format(np.amax(frac),cond))
    else:
        print("Good acceptance rate ({}) for condition {}".format(np.median(frac),cond))
        exit_code = 0
    return exit_code


def rel_error(a, b, thresh=1.0, name=""):
    rel_error = 100.0 * abs((a-b)/a)
    if rel_error > thresh:
        print("Discrepancy {} % between log-posterior prob.".format(rel_error)
            + " for estimate {}:".format(name))
        print("{} (computed) vs {} (closest sample)".format(a, b))
    return rel_error
