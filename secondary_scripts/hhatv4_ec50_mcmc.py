"""
Applying the same dose response curve fitting to our dataset (v4) of
HHAT-p8F-derived peptides that we applied to the MSKCC dose response,
for consistency of derived EC50s. We use the same Hill model function, with
parameters for the background and binomial error estimates (N_eff).
The Hill function describes the dose response measurement (fraction of
active cells) in linear scale (range 0-1), as well as the doses.
However, for fitting, the log of the EC50 parameter and the log of the
effective number of cells for error bars (N_eff) are fitted, so they have
a scale comparable to that of other parameters (V_inf, background, n).

We need to account for a few differences nonetheless:
    - MSKCC doses were in ug/mL, here we used doses in M.
    - MSKCC dose response had 4-1BB as the only activation marker; we
        measured several but in the end we use 4-1BB as well for consistency
    - MSKCC had only one T cell donor, we have 3-4, but only one (donor C)
    did not have technical issues during acquisition for surface markers,
    so in the end we have only 1 donor for 4-1BB.
    - MSKCC had three doses per peptide only, we have 6, so we need to
    scale up regularization factors accordingly.
    - Background activation seems higher in our measurements, so the parameter
    boundaries need to be adjusted.

As for MSKCC data, I use the MCMC (emcee package) to estimate more precisely
the posterior distribution of EC50 values for each peptide in the library.
I use multiprocessing to speed up the process. Results are saved using emcee's
default saving format (to avoid another complicated custom HDF5 file).

Warning: since there are over 200 dose response curves to fit via MCMC, 
this script can easily take 4-6 hrs to run on a multi-CPU cluster. 
Moreover, the full MCMC samples file is 1.5 GBs so we only
included the summary statistics (best fit, variance, CI, etc. of EC50 parameters)
in our data and results repository. The MCMC file is available upon request. 

@author: frbourassa
August 2024
"""

import numpy as np
import pandas as pd
import os
import sys
if "../" not in sys.path:
    sys.path.insert(0, "../")

from utils.cpu_affinity import count_parallel_cpu
# We reuse the MCMC launch and analysis functions defined for the MSKCC library
from secondary_scripts.mskcc_ec50_mcmc import (
    main_run_ec50_mcmc,
    main_analyze_ec50_mcmc,
    scatter_example_peptide,
    plot_samplecurves_peptide,
    plot_lsq_fits_peptide
)
n_cpu = count_parallel_cpu()

# For plotting
import matplotlib as mpl
# This means we are on the Physics servers
if sys.platform == 'linux':
    mpl.use("Agg")


def load_raw_data_hhatv4(fname=None, resp_name="41BB+"):
    if fname is None:
        fname = "../data/dose_response/hhatlibrary4_cellData.h5"
    df_raw_data = pd.read_hdf(fname)
    # Giving this TCR number 8, following the MSKCC count
    # Only donor C worked
    df_dose_hhat = pd.concat({"8":df_raw_data.xs("C", level="Donor")}, names=["TCR"])
    df_dose_hhat = pd.concat({"HHAT-L8F":df_dose_hhat}, names=["Antigen"])
    df_dose_hhat.index = df_dose_hhat.index.rename({"[Peptide]":"Dose_M"})
    df_dose_hhat = df_dose_hhat[resp_name].to_frame().unstack(["Dose_M"]) / 100.0
    df_dose_hhat[resp_name] = df_dose_hhat[resp_name].clip(0.0, 1.0)
    df_dose_hhat = df_dose_hhat.sort_index()
    return df_dose_hhat


if __name__ == "__main__":
    mcmc_filename = os.path.join("..", "results", "pep_libs", 
                                 "hhatv4_ec50_mcmc_results_backgnd.h5")
    lsq_filename = os.path.join("..", "results", "pep_libs", 
                                "hhatv4_ec50_lsq_results_backgnd.h5")
    stats_filename = os.path.join("..", "results", "pep_libs", 
                                  "hhatv4_ec50_mcmc_stats_backgnd.h5")

    # Selected examples, and controls from a previous peptide batch
    # The original sequence from which the library is derived is HHAT-L8F
    # and, WT is the self antigen HHAT
    pep_controls = ["p8F", "WildType", "DMSO"]  # p8F = previous batch of HHAT-p8F
    # L8F = new batch of HHAT-p8F in the ordered peptide library
    # WT = new batch of HHAT-WT in the ordered peptide library
    # (WT = reverse substitution F8L in the L8F peptide from which the library originated)
    pepexes = ["L8F", "WT", "L9C", "K1A", "V5I", "L7V"]
    pepexes = list(map(lambda x: ("HHAT-L8F", "8", x), pepexes+pep_controls))
    pepexes.remove(("HHAT-L8F", "8", "DMSO"))
    selection = None  #pepexes  # or None to run all

    # Parameter fitting arguments differing from those chosen to re-analyze
    # MSKCC dose response data
    # Add parameter for background activation: default is 0
	# Add parameter for binomial error model: effective number of cells
    # Prior: 50,000 cells (typical in our experiments).
    # Background is high in our experiments, but regularize to 0 by default
    # Regularize EC50s to be within 1e-8, 1e-5 typically.
    # Prior for logN_eff: one order of magnitude uncertainty,
    # so regul_rate = 0.5 * 1/1.0 = 0.5
    # Limit the background level to 0.1 and regularize strongly to be zero
    # Regul. rates for LSQ are the inverses of the variances of the priors
    # Limit Hill power to 2, N_eff to 10^6, EC50 to 0.1 M, background to 20 %
    # Amplitude and background regularization need to be adjusted to the
    # chosen response surface marker
    resp_name = "41BB+"
    default_ampli = 0.4 if resp_name == "CD25+" else 0.85
    max_back = 0.3 if resp_name == "41BB+" else 0.15
    fit_kwargs = {
        "n_boot": 1000,
        "thin_param": 10,
        "selection": selection,
        "data_loading_fct": load_raw_data_hhatv4,
        "dose_name": "Dose_M",
        "resp_name": resp_name,  # default: "41BB+"
        "p_names": ["V_inf", "n", "log_ec50_M", "logN_eff", "backgnd"],
        "p_defaults": np.asarray([default_ampli, 1.0, -9.0, -3.0, np.log10(5e4), 0.15]),
        "p_guess": np.asarray([0.5, 1.0, -7.0, np.log10(5e4), max_back/2.0]),
        "priors_stdevs": np.asarray([0.4, 1.0, 4.0, 2.0, 0.3]),
        "param_bounds":[np.asarray([max_back, 0.5, -12.0, 1.0, 0.0]),
                        np.asarray([1.0, 4.0, 0.0, 6.0, max_back])],
        "seedseq_main": np.random.SeedSequence(
            0x27991c04452980cd66e36bc0ae0f598b,
            spawn_key=(0x9e0f0af05887ad36825a6bc1d6598161,)
        ),
        "skip_mcmc": False
    }

    # Run the main fitting code
    if not os.path.isfile(mcmc_filename):
        print("Starting MCMC runs for EC50s of HHAT peptides...")
        main_run_ec50_mcmc(mcmc_filename, lsq_filename, **fit_kwargs)
    elif os.path.getsize(mcmc_filename) < 1e7:
        print("Starting MCMC runs for EC50s of HHAT peptides...")
        main_run_ec50_mcmc(mcmc_filename, lsq_filename, **fit_kwargs)
    else:
        print("Found existing MCMC results for HHAT peptides")


    # Summary statistics, just record standard deviation, assume normal
    lys_kwargs = {
        "selection": selection,
        "p_names": fit_kwargs.get("p_names"),
        "data_loading_fct": fit_kwargs.get("data_loading_fct"),
        "resp_name":fit_kwargs.get("resp_name")
    }
    if (not os.path.isfile(stats_filename)) and (os.path.isfile(mcmc_filename)):
        print("Starting analysis of MCMC runs...")
        main_analyze_ec50_mcmc(mcmc_filename, stats_filename, **lys_kwargs)
    else:
        print("Found existing MCMC analysis outputs for MSKCC peptides, "
                + "or did not find an MCMC results file. ")

    # Plot a few examples of MCMC samples and/or LSQ samples
    plot_kwargs = {
        "do_save": True,
        "p_names": fit_kwargs.get("p_names"),
        "data_loading_fct": load_raw_data_hhatv4,
        "dose_name": fit_kwargs.get("dose_name"),
        "resp_name": fit_kwargs.get("resp_name")
    }
    print("Starting plotting code")
    for pepex in pepexes:
        if os.path.isfile(mcmc_filename):
            plot_kwargs["fig_dir_prefix"] = "../figures/dose_response/hhatv4_mcmc/hhatv4_ec50_mcmc_corner_"
            scatter_example_peptide(
                mcmc_filename, lsq_filename, stats_filename, pepex, **plot_kwargs
            )
            plot_kwargs["fig_dir_prefix"] = "../figures/dose_response/hhatv4_mcmc/hhatv4_ec50_mcmc_fits_"
            plot_samplecurves_peptide(
                mcmc_filename, lsq_filename, stats_filename, pepex, **plot_kwargs
            )
        else:
            plot_kwargs["fig_dir_prefix"] = "../figures/dose_response/hhatv4_mcmc/hhatv4_ec50_lsq_fits_"
            plot_lsq_fits_peptide(lsq_filename, pepex, **plot_kwargs)
