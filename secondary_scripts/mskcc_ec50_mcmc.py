"""
The EC50 values provided in the Luksza et al., 2022, Nature paper do not
provide uncertainty estimates or confidence intervals, while they rely on
dose response curves comprising only 3 data points (for 3 parameters to fit).
Moreover, their fitting code is not available on Github and, due to least-squares
algorithm differences probably, I do not recover exactly the same EC50 values
as theirs when reanalyzing their dose response data with
scipy.optimize.least_squares. Importantly, the parameter covariance estimates
I obtain from the jacobian of the least-squares fit are very large, practically
unusable to propagate uncertainty in model antagonism predicitons, because
there are no error bars on the activation measurements, so the residuals
are not properly scaled.

So, here, I use MCMC (emcee package) to estimate more precisely the posterior
distribution of EC50 values for each peptide in the library. I use
multiprocessing to speed up the process. Results are saved using emcee's
default saving format (to avoid another complicated custom HDF5 file).

Warning: since there are 1200 dose response curves to fit via MCMC, 
this script can easily take 24hrs to run on a multi-CPU cluster. 
Moreover, the full MCMC samples files are several GBs, so we only
included the summary statistics (best fit, variance, CI, etc. of EC50 parameters)
in our data and results repository. They are available upon request. 

To fit and plot only a subset of peptides, change in the __main__ the variable 
`selection` variable from None to the list `pepexamples`. 

@author: frbourassa
July 2024
"""

import numpy as np
import pandas as pd
import scipy as sp
import os
import sys
if "../" not in sys.path:
    sys.path.insert(0, "../")


import emcee
from mcmc.mcmc_run import randomstate_from_rng
from multiprocessing import Pool
from utils.cpu_affinity import count_parallel_cpu
from time import perf_counter
from utils.fitting import (
    cramer_rao_fisher_pcov,
    resids_dose_hill,
    logpost_dose_hill_bounds_error_backgnd,
    resids_dose_hill_backgnd,
    cst_back_err
)
n_cpu = count_parallel_cpu()

# For plotting
import corner
import matplotlib.pyplot as plt
import matplotlib as mpl
# This means we are on the Physics servers
if sys.platform == 'linux':
    mpl.use("Agg")


### Data loading ###
def load_raw_data_mskcc(resp_name="Response (CD137+ fraction)"):
    df_raw_data_mskcc = pd.read_hdf(
        "../data/dose_response/MSKCC_rawDf.hdf", key="df")
    df_dose_mskcc = df_raw_data_mskcc.set_index(["Dose (ug/mL)"],
                    drop=True, append=True).drop("Dose (M)", axis=1)
    df_dose_mskcc = df_dose_mskcc.unstack(["Dose (ug/mL)"]) / 100.0
    df_dose_mskcc = df_dose_mskcc.rename(
        {"Response (CD137+ %)": resp_name}, axis=1)
    print(df_dose_mskcc[df_dose_mskcc[resp_name] > 1.0].dropna())
    df_dose_mskcc[resp_name] = df_dose_mskcc[resp_name].clip(0.0, 1.0)
    df_dose_mskcc = df_dose_mskcc.sort_index()
    return df_dose_mskcc


def get_tcr_to_ag_map(df):
    tcr_set = df.index.get_level_values("TCR").unique().sort_values()
    tcr_to_ag_map = {}
    for tcr in tcr_set:
        tcr_to_ag_map[tcr] = (df.xs(tcr, level="TCR").index
                                .get_level_values("Antigen")[0])
    return tcr_to_ag_map


### Main run and analysis functions ###
def main_run_ec50_mcmc(mcmc_file, lsq_file, **kwargs):
    """
    Keyword args:
        n_boot (int): default 1000
        thin_param (int): default 10
        selection (list or None): subset of peptides to run, run all if None
        data_loading_fct (callable): default load_raw_data_mskcc
        dose_name (str): default 'Dose (ug/mL)'
        resp_name (str): default "Response (CD137+ fraction)"
        p_defaults (np.ndarray): default [1.0, 1.0, -2.0, 2.0, np.log10(5e4), 0.0]
        p_guess (np.ndarray): default [1.0, 1.0, 0.0, log10(5e4), 0.0]
        priors_stdevs (np.ndarray): default [0.1, 1.0, 4.0, 3.0, 0.1]
        param_bounds (np.ndarray): default is:
                [np.asarray([0.15, 0.0, -4.0, 1.0, 0.0]),
                 np.asarray([1.0, 2.0, 5.0, 6.0, 0.15])]
        p_names (list of str): default is:
            ["V_inf", "n", "log_ec50_ugmL", "logN_eff", "backgnd"]
            Exactly one element should start with log_ec50.
        seedseq_main (np.random.SeedSequence): default has no seed
        skip_mcmc (bool): option to skip MCMC and do only lsq, for testing,
            default False.
    """
    lsq_file3 = lsq_file[:-3] + "_3.h5"
    lsq_file5 = lsq_file[:-3] + "_5.h5"

    # Get kwargs for dataset-specific or model-specific tunings
    selection = kwargs.get("selection", None)
    n_boot = kwargs.get("n_boot", 1000)
    thin_param = kwargs.get("thin_param", 10)
    data_loading_fct = kwargs.get("data_loading_fct", load_raw_data_mskcc)
    dose_name = kwargs.get("dose_name", "Dose (ug/mL)")
    resp_name = kwargs.get("resp_name", "Response (CD137+ fraction)")
    p_defaults = kwargs.get("p_defaults", None)  # update default below
    p_guess = kwargs.get("p_guess", None)  # update default below
    priors_stdevs = kwargs.get("priors_stdevs", None)  # update default below
    param_bounds = kwargs.get("param_bounds", None)  # update default below
    p_names = kwargs.get("p_names", None)  # update default below
    seedseq_main = kwargs.get("seedseq_main", np.random.SeedSequence())
    skip_mcmc = kwargs.get("skip_mcmc", False)

    # Data to fit
    df_dose = data_loading_fct(resp_name=resp_name)
    tcr_to_ag_map = get_tcr_to_ag_map(df_dose)
    print(tcr_to_ag_map)

    # Extract peptide doses (same for all peptides)
    x_doses = np.log10(df_dose.columns.get_level_values(dose_name)
                        .astype(float).values)
    print(x_doses)

    # Fit parameters
    # Add parameter for background activation: default is 0
	# Add parameter for binomial error model: effective number of cells
    # Prior: 50,000 cells (SI text, Luksza et al., 2022).
    if p_defaults is None:
        p_defaults = np.asarray([1.0, 1.0, -2.0, 2.0, np.log10(5e4), 0.0])
    if p_guess is None:
        p_guess = np.ones(5)  # V_inf, n, log_ec50, logN_eff, backgnd
        p_guess[2] = 0.0
        p_guess[4] = 0.0
        p_guess[3] = np.log10(5e4)
    # Prior for logN_eff: one order of magnitude uncertainty,
    # so regul_rate = 0.5 * 1/1.0 = 0.5
    # Limit the background level to 0.1 and regularize strongly to be zero
    # Regul. rates are the inverses of the variances of the priors
    if priors_stdevs is None:
        priors_stdevs = np.asarray([0.1, 1.0, 4.0, 3.0, 0.1])
    regul_rates_mcmc = 1.0 / priors_stdevs**2.0
    print("MCMC regularization rates:", regul_rates_mcmc)
    # Default regularization that should reproduce MSKCC's EC50s
    regul_rates_lsq3 = np.asarray([0.01, 0.01, 0.001])  * (x_doses.size / 3.0)

    # lowers, uppers. Limit Hill power to 2, N_eff to 10^6
    if param_bounds is None:
        param_bounds = [np.asarray([0.15, 0.0, -4.0, 1.0, 0.0]),
                        np.asarray([1.0, 2.0, 5.0, 6.0, 0.15])]
    param_bounds_3 = [a[:3] for a in param_bounds]
    # Prepare DataFrame containing fit results
    pep_index = df_dose.index.copy()
    if p_names is None:
        p_names = ["V_inf", "n", "log_ec50_ugmL", "logN_eff", "backgnd"]
        ec50_name = p_names[2]
    else:
        if "V_inf" not in p_names:
            raise ValueError("V_inf should be the amplitude parameter's name")
        where_ec50_name = [a.startswith("log_ec50") for a in p_names]
        if sum(where_ec50_name) != 1:
            raise ValueError("One parameter should start with log_ec50")
        else:
            ec50_name = p_names[where_ec50_name.index(True)]
    p_dev_names = list(map(lambda x: x + "_std", p_names))
    df_lsq3_results = pd.DataFrame(
        np.zeros([len(pep_index), 6]), index=pep_index,
        columns=pd.Index(p_names[:3] + p_dev_names[:3])
    )
    # Also do a least-squares for the full model with errors, should
    # give similar results to the MAP. Can then estimate parameter error
    # with the Hessian (Fisher information matrix) more accurately.
    df_lsq5_results = pd.DataFrame(
        np.zeros([len(pep_index), 10]), index=pep_index,
        columns=pd.Index(p_names + p_dev_names)
    )
    # MCMC parameters
    nwalkers = 32
    ndim = len(p_guess)
    rgen_main = np.random.default_rng(seedseq_main)
    # emcee's RandomState needs a RandomState.get_state() tuple to be
    # initialized, so generate these right away.
    # Can't use the cleaner, newer Numpy's SeedSequence.spawn structure 
    loop_index = df_dose.index if selection is None else selection
    pep_rs_seeds = {}
    for k in loop_index:
        pep_rs_seeds[k] = randomstate_from_rng(rgen_main)
    backend_filename = mcmc_file

    # Fit each peptide
    for k in loop_index:
        time_start = perf_counter()
        y_responses = df_dose.loc[k].values
        # Peptides with missing data
        if np.any(np.isnan(y_responses)):
            df_lsq3_results.loc[k] = np.nan  # NaN for most parameters
            # EC50 and its stdev will be determined later, from linear regression
            # of the max response versus EC50 of other peptides
            df_lsq3_results.loc[k, "V_inf"] = y_responses[~np.isnan(y_responses)].max()
            df_lsq3_results.loc[k, "V_inf_std"] = 0.0
            df_lsq5_results.loc[k] = np.nan
            df_lsq5_results.loc[k, "V_inf"] = y_responses[~np.isnan(y_responses)].max()
            df_lsq5_results.loc[k, "V_inf_std"] = 0.0
            if k[0] != "CMV":
                print("Peptide {} is unexpectedly missing data".format(k))
        # Peptides with 3 data points
        else:
            # Fit first three parameters with least-squares
            res = sp.optimize.least_squares(
                fun=resids_dose_hill,
                x0=p_guess[:3],
                args=(x_doses, y_responses, regul_rates_lsq3, p_defaults[:4]),
                bounds=param_bounds_3,
                method="trf",
            )
            if not res.success:
                raise RuntimeError("Optimal parameters not found: " + res.message)

            # Compute standard deviation from jacobian, only include jac
            # elements corresponding to fitted points
            pcov = cramer_rao_fisher_pcov(res)
            pbest = res.x
            if np.any(np.isnan(pcov)) or np.any(np.isnan(pbest)):
                print("Problem for", k, ", pbest:", pbest)
            # Clip EC50
            if dose_name == "Dose (ug/mL)":
                pbest[2] = np.clip(pbest[2], -4.0, 4.0)  # Clip between 1e-4, 1e4
            df_lsq3_results.loc[k, p_names[:3]] = pbest
            df_lsq3_results.loc[k, p_dev_names[:3]] = np.sqrt(pcov[np.diag_indices(pcov.shape[0])][:3])

            # Do full LSQ from there
            p_guess5 = p_guess.copy()
            p_guess5[:3] = pbest
            res = sp.optimize.least_squares(
                fun=resids_dose_hill_backgnd,
                x0=p_guess5,
                args=(x_doses, y_responses, regul_rates_mcmc, p_defaults),
                bounds=param_bounds,
                method="trf",
            )
            if not res.success:
                raise RuntimeError("Optimal parameters not found: " + res.message)
            # Compute standard deviation from jacobian, only include jac
            # elements corresponding to fitted points
            pcov5 = cramer_rao_fisher_pcov(res)
            pbest5 = res.x
            if np.any(np.isnan(pcov5)) or np.any(np.isnan(pbest5)):
                print("Problem for", k, ", pbest:", pbest5)
            df_lsq5_results.loc[k, p_names] = pbest5
            df_lsq5_results.loc[k, p_dev_names] = np.sqrt(pcov5[np.diag_indices(pcov5.shape[0])][:5])

            # With only 3 data points for 3 parameters, the covariance estimate is really bad
            # Instead do MCMC around the maximum likelihood, the regularization
            # is like a Gaussian prior already
            if not skip_mcmc:
                pos = np.tile(pbest5, (nwalkers, 1))
                # Use LSQ5 estimate as starting point
                pos = pbest5 + 1e-3 * rgen_main.normal(size=(nwalkers, len(p_names)))
                # Use initial guess for logN_eff, backgnd
                #pos[:, 3:5] = pos[:, 3:5] + 1e-3 * rgen_main.normal(size=[nwalkers, 2])

                if param_bounds != None:
                    pos = np.clip(pos, a_min=param_bounds[0]+1e-6,
                        a_max=param_bounds[1]-1e-6)
                state_init = emcee.State(pos, random_state=pep_rs_seeds[k])
                # Open new group in save file
                backend = emcee.backends.HDFBackend(backend_filename, name=str(k))
                backend.reset(nwalkers, ndim)
                with Pool(n_cpu) as pool:
                    sampler = emcee.EnsembleSampler(
                        nwalkers, ndim, logpost_dose_hill_bounds_error_backgnd,
                        args=(x_doses, y_responses, regul_rates_mcmc, p_defaults, param_bounds),
                        pool=pool, backend=backend
                    )
                    sampler.run_mcmc(state_init, n_boot,
                        thin_by=thin_param, progress=False);
            time_end = perf_counter()
            print("Time for peptide {}: {:.4f} s".format(k, time_end - time_start))
        continue

    # Compute some final results
    df_lsq3_results["K_a"] = 10.0**df_lsq3_results[ec50_name]
    df_lsq3_results.to_hdf(lsq_file3, key="df")
    df_lsq5_results["K_a"] = 10.0**df_lsq5_results[ec50_name]
    df_lsq5_results.to_hdf(lsq_file5, key="df")
    print(df_lsq5_results)


def analyze_pep_mcmc(mcmc_file, k):
    reader = emcee.backends.HDFBackend(mcmc_file, name=str(k), read_only=True)
    # Check convergence quickly
    tau = reader.get_autocorr_time(quiet=True);  # Keep going if chain too short
    run_size = reader.iteration
    tau[np.isnan(tau)] = run_size * 10
    run_size_taus = np.min(run_size / tau)
    #if run_size_taus < 50:
    #    print("Peptide {} has full run = {} taus".format(k, run_size_taus))
    burnin = max(0, int(2 * np.max(tau)))
    thin = max(1, int(0.5 * np.min(tau)))
    # Get relevant, independent samples
    samples = reader.get_chain(discard=burnin, flat=True, thin=thin)
    # From these, estimate variance of each parameter, assuming Gaussian
    stdev = np.std(samples, axis=0)
    # Find maximum a posteriori
    logprobs = reader.get_log_prob(discard=0, thin=1, flat=True)
    map_index = np.argmax(logprobs)
    map_prob = logprobs[map_index]
    map_estim = reader.get_chain(discard=0, thin=1, flat=True)[map_index]
    return stdev, map_estim, map_prob, tau


def main_analyze_ec50_mcmc(mcmc_file, stats_file, **kwargs):
    """
    Keyword args:
        selection (list or None): subset of peptides to run, run all if None
        data_loading_fct (callable): default load_raw_data_mskcc
        p_names (list of str): default is:
            ["V_inf", "n", "log_ec50_ugmL", "logN_eff", "backgnd"]
        resp_name (str): default "Response (CD137+ fraction)"
    """
    # Get kwargs for dataset-specific of model-specific tunings
    selection = kwargs.get("selection", None)
    data_loading_fct = kwargs.get("data_loading_fct", load_raw_data_mskcc)
    p_names = kwargs.get("p_names", None)
    resp_name = kwargs.get("resp_name", "Response (CD137+ fraction)")

    # load dataset, update missing keyword arguments
    if p_names is None:
        p_names = ["V_inf", "n", "log_ec50_ugmL", "logN_eff", "backgnd"]
    df_dose = data_loading_fct(resp_name=resp_name)

    df_summary_ec50 = pd.DataFrame(
        0.0,
        index=df_dose.index,
        columns=pd.MultiIndex.from_product(
            [["MAP", "stdev", "mcmc_tau"], p_names],
            names=["Feature", "Parameter"]
        )
    )
    # Add a column for log probabilities
    df_summary_ec50[("logprob", "all")] = 0.0

    # Analyze each peptide
    loop_index = df_dose.index if selection is None else selection
    for k in loop_index:
        time_start = perf_counter()
        y_responses = df_dose.loc[k].values
        # Peptides with missing data
        if not np.any(np.isnan(y_responses)):
            vari, map_estim, map_prob, taus = analyze_pep_mcmc(mcmc_file, k)
            df_summary_ec50.loc[k, "MAP"] = map_estim
            df_summary_ec50.loc[k, "stdev"] = vari
            df_summary_ec50.loc[k, ("logprob", "all")] = map_prob
            df_summary_ec50.loc[k, "mcmc_tau"] = taus
        else:
            df_summary_ec50.loc[k] = pd.NA  # Just no MCMC here
        time_end = perf_counter()
        print("Time to analyze peptide {}: {:.4f} s".format(k, time_end - time_start))

    # Save summary statistics to disk
    df_summary_ec50.to_hdf(stats_file, key="df")
    return df_summary_ec50


def scatter_example_peptide(mcmc_file, lsq_file, stats_file, k, **kwargs):
    """
    Keyword args:
        do_save (bool): default False
        fig_dir_prefix (str): path to the figures folder and prefix
            name of the figures; the default is:
            "../figures/dose_response/mskcc_mcmc/mskcc_ec50_mcmc_corner_"
    """
    # Get keyword arguments
    do_save = kwargs.get("do_save", False)
    fig_dir_prefix = kwargs.get("fig_dir_prefix",
        "../figures/dose_response/mskcc_mcmc/mskcc_ec50_mcmc_corner_")

    # Open the MCMC simulation
    reader = emcee.backends.HDFBackend(mcmc_file, name=str(k), read_only=True)
    # Check convergence quickly
    tau = reader.get_autocorr_time(quiet=True);  # Keep going if chain too short
    burnin = int(2 * np.max(tau))
    # Get relevant, independent samples
    samples = reader.get_chain(discard=burnin, flat=True)

    # Prepare plot
    lsq_file3 = lsq_file[:-3] + "_3.h5"
    lsq_file5 = lsq_file[:-3] + "_5.h5"
    df_lsq3 = pd.read_hdf(lsq_file3).loc[k]
    df_lsq5 = pd.read_hdf(lsq_file5).loc[k]
    df_stats = pd.read_hdf(stats_file).loc[k]
    labels = list(df_stats["MAP"].index.values)
    figure = corner.corner(samples, labels=labels)

    # Add np.nans to df_lsq3 for the uncertainty estimate and background,
    # so it does not have a crosshair for these parameters
    df_lsq3[labels[-1]] = np.nan  # logN_eff
    df_lsq3[labels[-2]] = np.nan  # backgnd
    # Add crosshairs
    corner.overplot_lines(figure, df_lsq3[labels].values, color="C0")
    corner.overplot_points(figure, df_lsq3[labels].values.reshape(1, -1),
        marker="s", color="C0")
    corner.overplot_lines(figure, df_lsq5[labels].values, color="C2")
    corner.overplot_points(figure, df_lsq5[labels].values.reshape(1, -1),
        marker="s", color="C2")
    corner.overplot_lines(figure, df_stats["MAP"].values, color="C1")
    corner.overplot_points(figure, df_stats["MAP"].values.reshape(1, -1),
        marker="o", color="C1")
    if do_save:
        figure.savefig(fig_dir_prefix + "_".join([str(a) for a in k]) + ".pdf",
            transparent=True, bbox_inches='tight')
        print("Finished plotting corner plot and saved {}".format(k))
    return figure


def mskcc_hill(log_doses, p):
    log_ec50 = p[2]
    hill_y = p[0] / (1.0 + 10.0 ** (p[1] * (log_ec50 - log_doses)))
    return hill_y

def hill_back(log_doses, p):
    log_ec50 = p[2]
    backgnd = p[4]
    hill_y = (p[0] - backgnd) / (1.0 + 10.0 ** (p[1] * (log_ec50 - log_doses)))
    hill_y += backgnd
    return hill_y


def plot_samplecurves_peptide(mcmc_file, lsq_file, stats_file, k, **kwargs):
    """
    Keyword args:
        do_save (bool): default False
        p_names (list): default
            ["V_inf", "n", "log_ec50_ugmL", "logN_eff", "backgnd"]
        data_loading_fct (callable): default load_raw_data_mskcc
        dose_name (str): default "Dose (ug/mL)"
        resp_name (str): default "Response (CD137+ fraction)"
        fig_dir_prefix (str): default
    """
    # Collect keyword arguments
    do_save = kwargs.get("do_save", False)
    p_names = kwargs.get("p_names",
        ["V_inf", "n", "log_ec50_ugmL", "logN_eff", "backgnd"])
    data_loading_fct = kwargs.get("data_loading_fct", load_raw_data_mskcc)
    dose_name = kwargs.get("dose_name", "Dose (ug/mL)")
    resp_name = kwargs.get("resp_name", "Response (CD137+ fraction)")
    fig_dir_prefix = kwargs.get("fig_dir_prefix",
        "../figures/dose_response/mskcc_mcmc/mskcc_ec50_mcmc_fits_")

    # Load least-squares results
    lsq_file3 = lsq_file[:-3] + "_3.h5"
    lsq_file5 = lsq_file[:-3] + "_5.h5"
    df_lsq3 = pd.read_hdf(lsq_file3).loc[k]
    df_lsq5 = pd.read_hdf(lsq_file5).loc[k]
    df_dose = data_loading_fct(resp_name=resp_name)

    # Load MCMC results
    reader = emcee.backends.HDFBackend(mcmc_file, name=str(k), read_only=True)
    # Check convergence quickly
    tau = reader.get_autocorr_time(quiet=True);  # Keep going if chain too short
    burnin = int(2 * np.max(tau))
    # Get relevant, independent samples
    all_samples = reader.get_chain(discard=burnin, flat=True)
    df_stats = pd.read_hdf(stats_file).loc[k]

    # Compute sample curves
    n_smp = 100
    ch_samples = all_samples[np.random.choice(all_samples.shape[0], size=n_smp)]
    min_conc = np.log10(df_dose.columns.get_level_values(dose_name).astype(float).min())
    max_conc = np.log10(df_dose.columns.get_level_values(dose_name).astype(float).max())
    log_doses = np.linspace(min_conc - 1.0, max_conc + 1.0, 200)
    param_order = p_names
    hill_map = mskcc_hill if "backgnd" not in param_order else hill_back
    lsq_curve3 = mskcc_hill(log_doses, df_lsq3[param_order[:3]].values)
    lsq_curve5 = hill_map(log_doses, df_lsq5[param_order].values)
    map_curve = hill_map(log_doses, df_stats.loc[("MAP", param_order)].values)
    sample_curves = np.zeros([n_smp, log_doses.shape[0]])
    for i in range(n_smp):
        sample_curves[i] = hill_map(log_doses, ch_samples[i])

    fig, ax = plt.subplots()
    # Plot data, lsq, then all sampled curves
    col_doses = df_dose.columns.get_level_values(dose_name)
    response_name = str(df_dose.columns.get_level_values(0)[0])
    data_pts_doses = np.log10(col_doses.astype(float).values)
    data_pts = df_dose.loc[k, (response_name, list(col_doses))]
    try:
        n_eff_idx = p_names.index("logN_eff")
    except IndexError:
        n_eff_map = 1e12  # not visible on the plot
    else:
        n_eff_map = 10.0**df_stats.at[("MAP", param_order[n_eff_idx])]
    data_pts_err = np.sqrt(data_pts * (1.0 - data_pts) / n_eff_map) + cst_back_err
    ax.errorbar(
        10.0**data_pts_doses, data_pts, yerr=data_pts_err, ecolor="C1",
        marker="o", ms=6, ls="none", mfc="k", mec="k", label="Data"
    )

    for i in range(n_smp):
        lbl = "MCMC sample" if i == 0 else None
        ax.plot(
            10.0**log_doses, sample_curves[i],
            color="grey", alpha=0.3, lw=1.0, label=lbl
        )
    ax.plot(10.0**log_doses, lsq_curve3, color="C0", ls="--", label="Least-squares 3-param")
    ax.plot(10.0**log_doses, lsq_curve5, color="C2", ls=":", label="Least-squares full")
    ax.plot(10.0**log_doses, map_curve, color="C1", ls="-", label="MAP")
    ax.set(xlabel="log10 " + dose_name.replace("_", " "), ylabel=response_name,
            xscale="log", title=str(k))
    ax.legend()
    fig.tight_layout()
    if do_save:
        fig.savefig(fig_dir_prefix + "_".join([str(a) for a in k]) + ".pdf",
            transparent=True, bbox_inches='tight')
        print("Finished plotting sample curves and saved {}".format(k))
    return fig, ax


def plot_lsq_fits_peptide(lsq_file, k, **kwargs):
    # Load kwargs
    do_save = kwargs.get("do_save", False)
    param_order = kwargs.get("p_names", None)
    dose_name = kwargs.get("dose_name", "Dose (ug/mL)")
    resp_name = kwargs.get("resp_name", "Response (CD137+ fraction)")
    fig_dir_prefix = kwargs.get("fig_dir_prefix",
        "../figures/dose_response/mskcc_mcmc/mskcc_ec50_mcmc_fits_")
    data_loading_fct = kwargs.get("data_loading_fct", load_raw_data_mskcc)

    # Load least-squares results
    lsq_file3 = lsq_file[:-3] + "_3.h5"
    lsq_file5 = lsq_file[:-3] + "_5.h5"
    df_lsq3 = pd.read_hdf(lsq_file3).loc[k]
    df_lsq5 = pd.read_hdf(lsq_file5).loc[k]
    df_dose = data_loading_fct(resp_name=resp_name)

    # Compute fitted curves
    n_smp = 100
    min_conc = np.log10(df_dose.columns.get_level_values(dose_name).astype(float).min())
    max_conc = np.log10(df_dose.columns.get_level_values(dose_name).astype(float).max())
    log_doses = np.linspace(min_conc - 1.0, max_conc + 1.0, 200)

    if param_order is None:
        param_order = ["V_inf", "n", "log_ec50_ugmL", "logN_eff", "backgnd"]
    hill_map = mskcc_hill if "backgnd" not in param_order else hill_back
    lsq_curve3 = mskcc_hill(log_doses, df_lsq3[param_order[:3]].values)
    lsq_curve5 = hill_map(log_doses, df_lsq5[param_order].values)

    fig, ax = plt.subplots()
    # Plot data, then lsq curves
    col_doses = df_dose.columns.get_level_values(dose_name)
    response_name = str(df_dose.columns.get_level_values(0)[0])
    data_pts_doses = np.log10(col_doses.astype(float).values)
    data_pts = df_dose.loc[k, (response_name, list(col_doses))]
    try:
        n_eff_map = 10.0**df_lsq5["logN_eff"]
        data_pts_err = np.sqrt(data_pts * (1.0 - data_pts) / n_eff_map) + cst_back_err
    except:
        data_pts_err = cst_back_err
    ax.errorbar(
        10.0**data_pts_doses, data_pts, yerr=data_pts_err, ecolor="C1",
        marker="o", ms=6, ls="none", mfc="k", mec="k", label="Data"
    )

    ax.plot(10.0**log_doses, lsq_curve3, color="C0", ls="--", label="Least-squares 3-param")
    ax.plot(10.0**log_doses, lsq_curve5, color="C2", ls=":", label="Least-squares full")
    ax.set(xlabel="log10 " + dose_name, ylabel=response_name,
            xscale="log", title=str(k))
    ax.legend()
    fig.tight_layout()
    if do_save:
        fig.savefig(fig_dir_prefix + "_".join([str(a) for a in k]) + ".pdf",
            transparent=True, bbox_inches='tight')
        print("Finished plotting least-squares curves and saved {}".format(k))
    return fig, ax


if __name__ == "__main__":
    mcmc_filename = os.path.join("..", "results", "pep_libs", 
                                 "mskcc_ec50_mcmc_results_backgnd.h5")
    lsq_filename = os.path.join("..", "results", "pep_libs", 
                                "mskcc_ec50_lsq_results_backgnd.h5")

    # Selected examples
    pepexamples = [
        ('CMV', '1', 'A7G'),
        #("CMV", "2", "V6D"),
        ("CMV", "1", "WT"),
        ("CMV", "2", "WT"),
        ("CMV", "3", "WT"),
        ('gp100', '6', 'V9C'),
        ('gp100', '6', 'V9S'),
        ('gp100', '6', 'P6R'),
        ("gp100", "4", "WT"),
        ("gp100", "5", "WT"),
        ("gp100", "6", "WT"),
        ("Neoantigen", "7", "WT"),
        ("Neoantigen", "7", "A5C"),
    ]
    selection = None  #pepexamples  # or None to run all

    # Fitting main function's kwargs
    fit_kwargs = {
        "n_boot": 1000,  # number of saved samples per peptide
        "thin_param": 10,  # save a sample every thin_param step 
        # (such that total steps taken = n_boot*thin_param)
        "selection": selection,
        "seedseq_main": np.random.SeedSequence(
            0x1e37eb788b7f4ffcb7f0535a05a1c89b,
            spawn_key=(0x83b7e436c442da36ab96de3ea80fda5,)
        ),
        "priors_stdevs": np.asarray([0.4, 1.0, 4.0, 2.0, 0.3]),  # Consistent with HHATv4
        "skip_mcmc": False
    }

    # Run the main fitting code
    if not os.path.isfile(mcmc_filename):
        print("Starting MCMC runs for EC50s of MSKCC peptides...")
        main_run_ec50_mcmc(mcmc_filename, lsq_filename, **fit_kwargs)
    elif os.path.getsize(mcmc_filename) < 1e8:
        print("Starting MCMC runs for EC50s of MSKCC peptides...")
        main_run_ec50_mcmc(mcmc_filename, lsq_filename, **fit_kwargs)
    else:
        print("Found existing MCMC results for MSKCC peptides")

    # Summary statistics, just record standard deviation, assume normal
    stats_filename = "../results/pep_libs/mskcc_ec50_mcmc_stats_backgnd.h5"
    if (not os.path.isfile(stats_filename)) and (os.path.isfile(mcmc_filename)):
        print("Starting analysis of MCMC runs...")
        main_analyze_ec50_mcmc(mcmc_filename, stats_filename, selection=selection)
    else:
        print("Found existing MCMC analysis outputs for MSKCC peptides")

    # Plot a few examples of MCMC samples
    print("Starting plotting code")
    plot_kwargs = {
        "do_save": True,
    }
    for pepex in pepexamples:
        if os.path.isfile(mcmc_filename):
            plot_kwargs["fig_dir_prefix"] = "../figures/dose_response/mskcc_mcmc/mskcc_ec50_mcmc_corner_"
            scatter_example_peptide(
                mcmc_filename, lsq_filename, stats_filename, pepex, **plot_kwargs
            )

            plot_kwargs["fig_dir_prefix"] = "../figures/dose_response/mskcc_mcmc/mskcc_ec50_mcmc_fits_"
            plot_samplecurves_peptide(
                mcmc_filename, lsq_filename, stats_filename, pepex, **plot_kwargs
            )
        else:
            plot_kwargs["fig_dir_prefix"] = "../figures/dose_response/mskcc_mcmc/mskcc_ec50_lsq_fits_"
            plot_lsq_fits_peptide(lsq_filename, pepex, **plot_kwargs)
