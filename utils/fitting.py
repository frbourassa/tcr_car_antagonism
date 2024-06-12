""" Module with functions to fit dose response curves, essentially.

@author: frbourassa
February 2024
Derived from code by soorajachar
"""

import numpy as np
import scipy as sp
from scipy import stats

def hillFunction(x, parameters, hill_k=1):
    amplitude = parameters[0]
    ec50 = parameters[1]
    background = parameters[2]
    return amplitude * x**hill_k/(ec50**hill_k + x**hill_k) + background


def hillFunction4p(x, params):
    amplitude = params[0]
    ec50 = params[1]
    background = params[2]
    hill_k = params[3]
    return amplitude * x**hill_k/(ec50**hill_k + x**hill_k) + background


# Function which computes the vector of residuals, with the signature fun(x, *args, **kwargs)
# Fitting Hill in log-log.
def cost_fit_hill4p(hill_pms, xpts, ypts, yerr, p0, reg_rate=0.2):
    """ p0: value around which to regularize each param. L1 regularization ="""
    # Compute Hill function at xpts
    y_fit = hillFunction4p(xpts, hill_pms)
    resids = (ypts - y_fit) / yerr
    # Add in L1 regularization
    regul = np.sqrt(reg_rate*np.abs(hill_pms - p0))
    resids = np.concatenate([resids, regul])
    return resids


def inverseHill(y, parameters, hill_k=1):
    amplitude=parameters[0]
    ec50=parameters[1]
    background=parameters[2]
    return ec50 * ((y - background) / (amplitude + background - y))**(1/hill_k)


def r_squared(xdata, ydata, func, popt, **f_kwargs):
    residuals = ydata - func(xdata, popt, **f_kwargs)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((ydata - np.mean(ydata))**2)
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared


def student_t_ci(pts, alpha=0.05):
    """ Compute confidence interval 1 - alpha for data points pts (1D array).
    Can be applied on Dataframe using groupby. """
    n = len(pts)
    if n < 2: return np.nan
    y_mean = np.mean(pts)
    y_std = np.std(pts, ddof=1)  # Sample variance, divide by n-1
    t_crit = sp.stats.t.ppf(1.0 - alpha/2, df=n-1)
    # Scaled variable (\bar{X} - mu) / (std / sqrt(n)) follows Student t
    # Compute pt where its cdf is at 1 - alpha/2, then transform to isolate mu.
    # Interval (range to add/remove to mean estimator): t_crit * sqrt / sqrt(N)
    interv = t_crit * y_std / np.sqrt(n)

    return interv

# Utility function to find appropriate bounds on min of a quantity z
# when we don't know the sign of that quantity.
def find_bounds_on_min(z, lo=0.25, hi=2.0):
    if np.min(z) < 0:
        min_z = np.min(z) * hi
        max_z = min_z * lo
    elif np.min(z) == 0.0:
        if np.max(z) > 0.0:
            max_z = np.quantile(z[np.nonzero(z)], 0.1)
        else:
            max_z = 0.01
        min_z = 0.0
    else:
        min_z = np.min(z) / hi
        max_z = min_z / lo
    return min_z, max_z


def linear_error_from_log_bar(y, bar, base=10.0):
    y_up = 10**(y + bar)
    y_lo = 10**(y - bar)
    yerr = np.stack([10**y - y_lo, y_up - 10**y], axis=0)
    return yerr


#Function which computes the vector of residuals, with the signature fun(x, *args, **kwargs)
# Fitting Hill in log-log.
def cost_fit_hill(hill_pms, xpts, ypts, yerr, p0, hill_k=1, reg_rate=0.2):
    """ p0: value around which to regularize each param. L1 regularization ="""
    # Compute Hill function at xpts
    y_fit = hillFunction(xpts, hill_pms, hill_k)
    resids = (ypts - y_fit) / yerr
    # Add in L1 regularization
    regul = np.sqrt(reg_rate*np.abs(hill_pms - p0))
    resids = np.concatenate([resids, regul])
    return resids


def cramer_rao_fisher_pcov(res):
    """ From least-squares fit results, estimate parameter covariance
    using the Cramér-Rao bound and the Fisher information matrix approximated
    as the curvature of the least-squares cost function
    """
    # Fisher information estimate of the error on parameters: inverse of hessian
    # Uses the Cramér-Rao bound: Var[x] \geq 1 / Fisher[x]
    # Fisher = d^2/dtheta^2[log f(theta)], where f = likelihood = gaussian
    # so log f is just the sum of squares, our cost function here
    # so this is just the hessian of the least-squares cost.
    # Can approximate the hessian as J^T J with J the jacobian
    # Hence, we just need here to invert the jacobian
    # Do Moore-Penrose inverse discarding zero singular values.
    _, s, vt = np.linalg.svd(res.jac, full_matrices=False)
    threshold = np.finfo(float).eps * max(res.jac.shape) * s[0]
    s = s[s > threshold]
    vt = vt[:s.size]
    pcov = np.dot(vt.T / s**2, vt)
    # Always assuming sigmas are absolute, so no need to rescale this covariance
    #if not absolute_sigma:
    #    if ysize > p0.size:
    #        s_sq = cost / (ysize - p0.size)
    #        pcov = pcov * s_sq
    #    else:
    #        pcov.fill(inf)
    #        warn_cov = True
    return pcov
