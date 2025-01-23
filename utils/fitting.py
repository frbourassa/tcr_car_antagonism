""" Module with functions to fit dose response curves, essentially.

@author: frbourassa
February 2024
Derived from code by soorajachar
"""

import numpy as np
import scipy as sp
import math

# Avoid excessively small errors by adding a small positive number
# to the binomial error model on cell fractions.
# 0.02 is reasonable: corresponds to error of 2 % active cells
# surely the technical noise is larger than this
cst_back_err = 0.04

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
    if ss_tot != 0.0:
        r_squared = 1 - (ss_res / ss_tot)
    elif ss_res == 0.0:  # All ydata identical and we fit them perfectly
        r_squared = 1.0
    else:  # Imperfect fit on constant ydata
        r_squared = -np.inf
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


def resids_dose_hill(p, log_doses, responses, reg_rates, p_targets):
    """ residuals function for the regularized Hill fit.
    The cost computed by least_squares is sum_i (resids[i])**2 / 2
    So for regularization, provide param - target

    Args:
        p (np.ndarray): vector of Hill fit parameters,
            V_inf
            n
            log_ec50
        log_doses (np.ndarray): x coordinates of the dose response points
        responses (np.ndarray): y coordinates of the dose response points
        reg_rates (np.ndarray): r_v, r_n, r_ec50. Will use their sqrt
        p_targets (np.ndarray): [V_inf, n, log_ec50low, log_ec50hi]

    Returns:
        resids (np.ndarray): difference between fit and data
            at each point, plus difference between parameter and target
            (i.e. square root of regularization term).

    """
    log_ec50 = p[2]
    hill_y = p[0] / (1.0 + 10.0 ** (p[1] * (log_ec50 - log_doses)))
    # d(EC50) = max(0; log(EC50)-log(100); log(0.01) - log(EC50)).
    reg_ec50 = max(0.0, log_ec50 - p_targets[3], p_targets[2] - log_ec50)
    resids = np.concatenate([
        hill_y - responses,
        np.sqrt(reg_rates[:2])*(p[:2] - p_targets[:2]),
        np.asarray([math.sqrt(reg_rates[2]) * reg_ec50])
    ])
    return resids


def sparsity_hill(x_doses, p0):
    n_x = x_doses.size
    js = np.zeros([n_x + len(p0), n_x], dtype=int)
    js[:n_x, :n_x] = 1
    i = np.arange(n_x, n_x + len(p0))
    js[i, np.arange(n_x)] = 1
    return js

def logpost_dose_hill(*args):
    return -0.5 * np.sum(resids_dose_hill(*args)**2)


def logpost_dose_hill_bounds(*args):
    """ last argument is pbounds on V_inf, n, log_ec50_ugmL,
    passed as two arrays of lows and highs """
    p = args[0]
    pbounds = args[-1]
    if np.any(p < pbounds[0]) or np.any(p > pbounds[1]):
        return -np.inf
    else:
        return -0.5 * np.sum(resids_dose_hill(*args[:-1])**2)


# Important: MCMC requires error bars on data points, we have none.
# So need to add sigma of each data point as a parameter.
# Assume same error on all points: one extra MCMC parameter to fit.
def logpost_dose_hill_bounds_error(
        p, log_doses, responses, reg_rates, p_targets, p_bounds
    ):
    r""" Same as logpost_dose_hill_bounds but with an extra
    parameter to account for the unknown uncertainty on the dose
    response points.

    Use a binomial model for the uncertainty on the fraction f, assuming
    N cells (unknown number) were measured in total. The true fraction
    of cells that should be active in a given condition, if many repeats,
    is f. Then the data is an estimate of f, \hat{f}, obtained by measuring
    some (variable) number of active cells, X. X follows a binomial(N, f),
    and thus has mean Nf and variance Nf(1-f).
    The estimate (data) is \hat{f} = X/N, the variance on this estimate
    is Var[\hat{f}] = Var[X/N] = Var[X]/N^2 = f(1-f)/N.
    Given only the data \hat{f}, its error bar (stdev, standard error)
    is therefore approximated by \sqrt{ \hat{f}(1-\hat{f}) / N }

    Thus, the parameter that we need to fit is N, the number of cells
    measured on average per dose response data point.

    Actually, this number
    is provided in the SI: N = 5x10^4 T cells, so 1/sqrt(N) = 0.01.
    Error bars would thus be really small -- too small.
    We need to add another element to the uncertainty model: constant sigma
    coming from biological variability. Assume this is also binomial,
    so uncertainties are smaller for definitely 0 or 100 % activation.
    We are therefore fitting an effective N for the uncertainty by MCMC.

    By default, use 50,000 as the prior (from the paper), but this will
    probably be larger.

    Args:
        p: last element is guess for standard deviation
            (relative to response fraction [0, 1])
        reg_rates: last element is the regularization rate on the
            logN_eff parameter
        p_targets: last element is the default logN_eff we assume a priori
        p_bounds: last element of each array is the lower or upper bound on
            logN_eff is bounds on logN_eff.
    """
    # Prior, part 1: boundaries
    if np.any(p < p_bounds[0]) or np.any(p > p_bounds[1]):
        return -np.inf

    # Prior, part 2: regularization = Gaussian priors within boundaries
    # d(EC50) = max(0; log(EC50)-log(100); log(0:01) - log(EC50)).
    log_ec50 = p[2]
    logn_eff = p[3]
    reg_ec50 = max(0.0, log_ec50 - p_targets[3], p_targets[2] - log_ec50)
    regul = np.concatenate([
        np.sqrt(reg_rates[:2])*(p[:2] - p_targets[:2]),
        np.asarray([
            math.sqrt(reg_rates[2]) * reg_ec50,
            math.sqrt(reg_rates[3]) * (p[3] - p_targets[4])])
        ])
    logprior = -0.5 * np.sum(regul**2.0)

    # Error model: binomial with effective number of cells, prior = 50k
    # used in the experiment
    n_eff = 10.0 ** logn_eff
    errors = np.sqrt(responses * (1.0 - responses) / n_eff) + cst_back_err

    # Likelihood: compute residuals with data, using sigma parameter
    hill_y = p[0] / (1.0 + 10.0 ** (p[1] * (log_ec50 - log_doses)))
    resids = (hill_y - responses) / errors

    loglikelihood = -0.5 * np.sum(resids**2.0)
    # The log of a Gaussian with unknown uncertainties should contain these
    # uncertainties, -log(1/sqrt(2*pi*sigma)) = -\sum_i 0.5*log(2*pi\sigma_i^2)
    # errors is already sigma, not sigma^2 and we can drop the 2*pi,
    # irrelevant constants.
    # Maximize log-likelihood: minimize errors, maximize N_eff. Consistent.
    loglikelihood -= np.log(errors).sum()

    # Bayes rule to combine likelihood and log-prior
    return loglikelihood + logprior

def logpost_dose_hill_bounds_csterror(
        p, log_doses, responses, reg_rates, p_targets, p_bounds
    ):
    """ Same as logpost_dose_hill_bounds but with an extra
    parameter to account for the unknown uncertainty on the dose
    response points.

    Use a constant standard deviation for all data points, so one
    extra parameter to fit only. The binomial error model is too clever
    for the amount of data we have, it leads to spurious fits for low
    amplitude dose response curves being fit to a low EC50 (so these
    peptides seem strong while they in fact produce no response).

    Use a prior on the order of 10 % of active cells, i.e. std = 0.1.

    Args:
        p: last element is guess for standard deviation
            (relative to response fraction [0, 1])
        reg_rates: last element is the regularization rate on the
            sigma parameter
        p_targets: last element is the default sigma we assume a priori
        p_bounds: last element of each array is the lower or upper bound on
            sigma is bounds on sigma.
    """
    # Prior, part 1: boundaries
    if np.any(p < p_bounds[0]) or np.any(p > p_bounds[1]):
        return -np.inf

    # Prior, part 2: regularization = Gaussian priors within boundaries
    # d(EC50) = max(0; log(EC50)-log(100); log(0:01) - log(EC50)).
    log_ec50 = p[2]
    sigma = p[3]
    reg_ec50 = max(0.0, log_ec50 - p_targets[3], p_targets[2] - log_ec50)
    regul = np.concatenate([
        np.sqrt(reg_rates[:2])*(p[:2] - p_targets[:2]),
        np.asarray([
            math.sqrt(reg_rates[2]) * reg_ec50,
            math.sqrt(reg_rates[3]) * (sigma - p_targets[4])])
        ])
    logprior = -0.5 * np.sum(regul**2.0) / sigma**2

    # Error model: constant standard deviation sigma
    # Avoid excessively small errors by adding small positive number
    # 1e-4 is reasonable: corresponds to error of 0.01 % active cells

    # Likelihood: compute residuals with data, using sigma parameter
    hill_y = p[0] / (1.0 + 10.0 ** (p[1] * (log_ec50 - log_doses)))
    resids = (hill_y - responses) / sigma

    loglikelihood = -0.5 * np.sum(resids**2.0)
    # The log of a Gaussian with unknown uncertainties should contain these
    # uncertainties, -log(1/sqrt(2*pi*sigma)) = -\sum_i 0.5*log(2*pi\sigma_i^2)
    n_data = resids.size
    loglikelihood -= np.log(sigma) * n_data

    # Bayes rule to combine likelihood and log-prior
    return loglikelihood + logprior


### Model with N_eff for error bars and a background parameter
def resids_dose_hill_backgnd(p, log_doses, responses, reg_rates, p_targets):
    """ residuals function for the regularized Hill fit, with background
    and error bar parameters.
    The cost computed by least_squares is sum_i (resids[i])**2 / 2
    So for regularization, provide param - target

    See logpost_dose_hill_bounds_error_backgnd for details on the parameters

    Args:
        p (np.ndarray): vector of Hill fit parameters,
            V_inf, n, log_ec50, logN_eff, backgnd
        log_doses (np.ndarray): x coordinates of the dose response points
        responses (np.ndarray): y coordinates of the dose response points
        reg_rates (np.ndarray): r_v, r_n, r_ec50, r_logN_eff, r_backgnd.
            Will use their sqrt.
        p_targets (np.ndarray): [V_inf, n, log_ec50low, log_ec50hi,
                                logN_eff, backgnd]

    Returns:
        resids_with_regul (np.ndarray): difference between fit and data
            at each point, plus difference between parameter and target
            (i.e. square root of regularization term).

    """
    log_ec50 = p[2]
    logn_eff = p[3]
    backgnd = p[4]
    reg_ec50 = max(0.0, log_ec50 - p_targets[3], p_targets[2] - log_ec50)
    # Error model: binomial with effective number of cells, prior = 50k
    # used in the experiment
    n_eff = 10.0 ** logn_eff
    errors = np.sqrt(responses * (1.0 - responses) / n_eff) + cst_back_err

    # Likelihood: compute residuals with data, using sigma parameter
    hill_y =  (p[0] - backgnd) / (1.0 + 10.0 ** (p[1] * (log_ec50 - log_doses))) + backgnd
    resids = (hill_y - responses) / errors

    # Add residuals dimensions for parameter regularizaton
    resids_with_regul = np.concatenate([
        resids,
        np.sqrt(reg_rates[:2])*(p[:2] - p_targets[:2]),
        np.asarray([
            math.sqrt(reg_rates[2]) * reg_ec50,
            math.sqrt(reg_rates[3]) * (logn_eff - p_targets[4]),
            math.sqrt(reg_rates[4]) * (backgnd - p_targets[5])
        ]),
    ])
    return resids_with_regul


def logpost_dose_hill_bounds_error_backgnd(
        p, log_doses, responses, reg_rates, p_targets, p_bounds
    ):
    r""" Same as logpost_dose_hill_bounds but with an extra
    parameter to account for the unknown uncertainty on the dose
    response points, and another to account for background activation.

    Use a binomial model for the uncertainty on the fraction f, assuming
    N cells (unknown number) were measured in total. The true fraction
    of cells that should be active in a given condition, if many repeats,
    is f. Then the data is an estimate of f, \hat{f}, obtained by measuring
    some (variable) number of active cells, X. X follows a binomial(N, f),
    and thus has mean Nf and variance Nf(1-f).
    The estimate (data) is \hat{f} = X/N, the variance on this estimate
    is Var[\hat{f}] = Var[X/N] = Var[X]/N^2 = f(1-f)/N.
    Given only the data \hat{f}, its error bar (stdev, standard error)
    is therefore approximated by \sqrt{ \hat{f}(1-\hat{f}) / N }

    Thus, the parameter that we need to fit is N, the number of cells
    measured on average per dose response data point.

    Actually, this number
    is provided in the SI: N = 5x10^4 T cells, so 1/sqrt(N) = 0.01.
    Error bars would thus be really small -- too small.
    We need to add another element to the uncertainty model: constant sigma
    coming from biological variability. Assume this is also binomial,
    so uncertainties are smaller for definitely 0 or 100 % activation.
    We are therefore fitting an effective N for the uncertainty by MCMC.

    By default, use 50,000 as the prior (from the paper), but this will
    probably be larger.

    Args:
        p: last element is guess for standard deviation
            (relative to response fraction [0, 1])
        reg_rates: last element is the regularization rate on the
            logN_eff parameter
        p_targets: last element is the default logN_eff we assume a priori
        p_bounds: last element of each array is the lower or upper bound on
            logN_eff is bounds on logN_eff.
    """
    # Prior, part 1: boundaries
    if np.any(p < p_bounds[0]) or np.any(p > p_bounds[1]):
        return -np.inf
    # Parameter order: V_inf, n, log_ec50, logN_eff, backgnd
    # Prior, part 2: regularization = Gaussian priors within boundaries
    # d(EC50) = max(0; log(EC50)-log(100); log(0:01) - log(EC50)).
    log_ec50 = p[2]
    logn_eff = p[3]
    backgnd = p[4]
    reg_ec50 = max(0.0, log_ec50 - p_targets[3], p_targets[2] - log_ec50)
    regul = np.concatenate([
        np.sqrt(reg_rates[:2])*(p[:2] - p_targets[:2]),
        np.asarray([
            math.sqrt(reg_rates[2]) * reg_ec50,
            math.sqrt(reg_rates[3]) * (logn_eff - p_targets[4]),
            math.sqrt(reg_rates[4]) * (backgnd - p_targets[5])
        ]),
    ])
    logprior = -0.5 * np.sum(regul**2.0)

    # Error model: binomial with effective number of cells, prior = 50k
    # used in the experiment
    n_eff = 10.0 ** logn_eff
    errors = np.sqrt(responses * (1.0 - responses) / n_eff) + cst_back_err

    # Likelihood: compute residuals with data, using sigma parameter
    hill_y =  (p[0] - backgnd) / (1.0 + 10.0 ** (p[1] * (log_ec50 - log_doses))) + backgnd
    resids = (hill_y - responses) / errors

    loglikelihood = -0.5 * np.sum(resids**2.0)
    # The log of a Gaussian with unknown uncertainties should contain these
    # uncertainties, -log(1/sqrt(2*pi*sigma)) = -\sum_i 0.5*log(2*pi\sigma_i^2)
    # errors is already sigma, not sigma^2 and we can drop the 2*pi,
    # irrelevant constants.
    # Maximize log-likelihood: minimize errors, maximize N_eff. Consistent.
    loglikelihood -= np.log(errors).sum()

    # Bayes rule to combine likelihood and log-prior
    return loglikelihood + logprior


def hill_with_back_diff(log_doses, p):
    backgnd = p[3]
    log_ec50 = p[2]
    hill_y = (p[0] - backgnd) / (1.0 + 10.0 ** (p[1] * (log_ec50 - log_doses))) + backgnd
    return hill_y

## FITTING AND PREDICTION
# Confidence interval and prediction interval for linear regression
# See Scott, 2020, Statistics: A Concise Mathematical Introduction for Students, Scientists, and Engineers
# section 8.5. The intervals are based on the assumption that y follows a normal
# distribution at X=x, with mean mx+b,
# hence y-(mx+b) / (variance estimator) follows a Student-t distribution.

# There is a difference between predicting the mean and its estimation error (confidence interval)
# and predicting an actual new sampled value of y at X=x (prediction interval).
# For the regression on model output vs survival, the relevant interval is the confidence interval;
# For the prediction of survival in new conditions, the relevant interval is the prediction interval.

# 95 % confidence interval for Student-t: alpha=0.05, t_crit = value of T for which cdf is 1-alpha/2.
# Inverse of cdf: ppf (Percent point function)

# And now a function to return the confidence interval on the regression mean at each x on a range
# Return the upper and lower y limits
# This was already available in statsmodel but I wanted to understand:
# https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLSResults.conf_int.html
def student_t_confid_interval_linregress(x, alpha, ndf, mean_x_estim, vari_estim, ssx, slope, intercept):
    """ Compute the confidence interval (upper and lower y values)
    of a linear regression at values x. This gives the region in which there
    is a probability (1-alpha) to find the true mean of the linear regression.

    Args:
        x (np.ndarray): values of X at which the interval will be evaluated
        alpha (float): 1 - confidence level (e.g. alpha=0.05 for 95 % ci)
        ndf (int): number of degrees of freedom = nb data points - nb parameters
        mean_x_estim (float): sample mean of X values on which the regression
            was performed, i.e. x_bar = 1/n_pts \sum_i x_i
        vari_estim (float): sample variance of residuals,
            \sum_i (y_i - (mx_i +b))**2 / (n_pts-2)
        ssx (float): \sum_i (x_i - mean_x_estim)**2
        slope (float): slope of the regression, i.e., m in mx + b
        intercept (float): y-axis intercet of the regression, i.e., b in mx + b

    Returns:
        y_lo (np.ndarray): lower y values of the confidence interval at each x
        y_up (np.ndarray): upper y values of the confidence interval at each x
    """
    t_crit = sp.stats.t.ppf(q=1.0 - alpha/2.0, df=ndf-2)
    abs_range = t_crit * np.sqrt(vari_estim * (1.0/ndf + (x - mean_x_estim)**2/ssx))
    return abs_range

def student_t_predict_interval_linregress(x, alpha, ndf, mean_x_estim, vari_estim, ssx, slope, intercept):
    r""" Compute the prediction interval (upper and lower y values)
    of a linear regression at values x. This is larger than the confidence interval;
    it gives the region in which there is a probability (1-alpha) to find a new
    value not included in the regression fit.

    Args:
        x (np.ndarray): values of X at which the interval will be evaluated
        alpha (float): 1 - confidence level (e.g. alpha=0.05 for 95 % ci)
        ndf (int): number of degrees of freedom = nb data points - nb parameters
        mean_x_estim (float): sample mean of X values on which the regression
            was performed, i.e. x_bar = 1/n_pts \sum_i x_i
        vari_estim (float): sample variance of residuals,
            \sum_i (y_i - (mx_i +b))**2 / (n_pts-2)
        ssx (float): \sum_i (x_i - mean_x_estim)**2
        slope (float): slope of the regression, i.e., m in mx + b
        intercept (float): y-axis intercet of the regression, i.e., b in mx + b

    Returns:
        y_lo (np.ndarray): lower y values of the prediction interval at each x
        y_up (np.ndarray): upper y values of the prediction interval at each x
    """
    t_crit = sp.stats.t.ppf(q=1.0 - alpha/2.0, df=ndf-2)
    abs_range = t_crit * np.sqrt(vari_estim * (1.0 + 1.0/ndf + (x - mean_x_estim)**2/ssx))
    return abs_range
