r"""
Fit recent TCR-TCR antagonism ratio data for 6F T cells
(mutated Y->F on the six CD3z ITAMs). These cells have different
KPR and inhibitory module activation properties (Gaud et al., Nat Immunol, 2023)
so the model parameters have to be re-estimated separately for them (instead
of enforcing parameter modification by hand, use MCMC for unbiased fit).

Setting N=4 for these cells, they have poorer ligand discrimination abilities
and fewer ITAM phosphorylation steps at the beginning of the cascade.
Expecting m=1 is best (seeing that E1 is the best antagonist for them).

Only using the updated AKPR model (inhibitory module, nonlinear psi)
since it was shown to be better than the SHP-1 model already.

Grid search integer parameters $m$, $f$, $k_I$
and for each, optimize $C_{m, thresh}$, $S_0$, and $\psi_0$.
It would be fair to change m, f, k_I but if many choices are very similar,
prioritize the same choices as for 6Y TCRs (except maybe m).

@author: frbourassa
March 2023
"""

import numpy as np
import pandas as pd
import h5py
from time import perf_counter

# Local modules
import sys, os
if not "../" in sys.path:
    sys.path.insert(1, "../")

from mcmc.costs_tcr_tcr_antagonism import cost_antagonism_akpr_i
from mcmc.mcmc_run import grid_search_emcee
from mcmc.utilities_tcr_tcr_antagonism import (
    prepare_data_6f,
    load_tcr_tcr_molec_numbers
)
from utils.preprocess import time_dd_hh_mm_ss


def main_tcr_tcr_run_6f(data_list, mm_params, data_fname, **kwargs):
    """
    kwargs:
        R_tot
        n_samp
        thin_by
        fit_bounds
        seed_sequence
        cost_fct
        prior_dist
        results_file: path to HDF5 file where results will be saved
    """
    # Default simlulation step numbers
    n_samp = kwargs.get("n_samp", 10000)
    thin_by = kwargs.get("thin_by", 4)
    cost_fct = kwargs.get("cost_fct", cost_antagonism_akpr_i)
    df_fit, df_ci_log2, tau_agonist = data_list
    # Model parameters that will remain fixed
    kappa = 1e-4
    # Normalized inhibitory species amount
    I_tot = 1.0

    # Number of TCR: based on calibration data, passed as argument R_tot
    R_tot = kwargs.get("R_tot", 100000)  # Default of 100k TCRs/cell

    # Number of steps, etc.
    N = 4
    
    # Fitted parameters, typical values: phi=0.1, 
    # C_m_thresh=300, I_thresh=0.5, psi_0=0.01
    # Parameters gridded over: k_I, m, f

    fit_param_names = [r"$\varphi^T$", r"$C^T_{m,th}$",
                    r"$I^T_{th}$", r"$\psi^T_0$"]
    # Bounds on parameters: phi, cmthresh, sthresh, psi_0
    # Use relatively tight bounds based on previous runs.
    # Use the log of parameters so the MCMC steps are even in log scale
    fit_bounds = kwargs.get("fit_bounds", None)
    prior_dist = kwargs.get("prior_dist", "uniform")
    if fit_bounds is None and prior_dist == "uniform":
        fit_bounds = [(0.05, 1.0), (1.0, 100*R_tot), (1e-5, 1000*I_tot), (1e-8, 1e-1)]
        fit_bounds = [np.log10(np.asarray(a)) for a in zip(*fit_bounds)]

    # Wrapping up parameters
    rates_others = [kappa]  # k_I will be gridded over
    total_RI = [R_tot, I_tot]

    # Fit
    # Grid search over m, f, k_I, each time using MCMC
    # to adjust C_m threshold, I threshold, and psi_0
    # Do not include k=2 as we know this doesn't work well on 6Y,
    # the 6F is even simpler so k=2 would be hard to motivate,
    # and it seems to lead to overfitting when making TCR/CAR predictions
    kmf_bounds = [(1, 2), (1, 4), (1, 2)]
    n_grid_pts = np.prod([a[1]-a[0]+1 for a in kmf_bounds])
    nwalkers = 32

    seed_sequence = kwargs.get("seed_sequence", None)
    if seed_sequence is None:
        seed_sequence = np.random.SeedSequence(
            0x92ea74f33e7a98edefc0c171aee692f1,
            spawn_key=(0xc2b221db89ad05809ce1125db6515652,)
        )
    # Skip computing MI in the cost function
    cost_args = (rates_others, total_RI, N, tau_agonist, df_fit, df_ci_log2)
    # List only the names of cost function args to save to the hdf5 file.
    cost_args_names = ["rates_others", "total_RI", "N", "tau_agonist"]
    results_file = kwargs.get("results_file", 
        os.path.join("..", "results", "mcmc", "mcmc_results_tcr_tcr_6f.h5")
    )
    try:
        start_t = perf_counter()
        grid_search_emcee(cost_fct,
            kmf_bounds, fit_bounds, results_file, p0=None, nwalkers=nwalkers,
            nsamples=n_samp, seed_sequence=seed_sequence, cost_args=cost_args,
            cost_kwargs={}, emcee_kwargs=kwargs.get("emcee_kwargs", {}),
            run_kwargs={"tune":True, "thin_by":thin_by},
            param_names=fit_param_names, prior_dist=prior_dist)
    except Exception as e:
        os.remove(results_file)
        print("Carried exception over")
        print(e)
    else:
        print("Finished grid search without uncontrolled exception")
        end_t = perf_counter()
        delta_t = end_t - start_t
        nsteps = nwalkers * n_samp * thin_by * n_grid_pts
        print("Total run time for MCMC: {} s".format(time_dd_hh_mm_ss(delta_t)))
        print("Time per step: {} s".format(delta_t / nsteps))
        # Add the cost function args to the results file, except the data.
        # Add the name of the data file, though.
        results_obj = h5py.File(results_file, "r+")
        args_group = results_obj.create_group("data")
        args_group.attrs["cost_args_names"] = cost_args_names
        args_group.attrs["data_file_name"] = data_fname  # full path, really
        args_group.attrs["thin_by"] = thin_by
        args_group.attrs["run_time"] = delta_t
        args_group.attrs["nsteps"] = nsteps
        for i in range(len(cost_args_names)):
            args_group.create_dataset(cost_args_names[i], data=cost_args[i])
        # Add the L-pulse conversion parameters too.
        args_group.create_dataset("l_conc_mm_params", data=mm_params)
        results_obj.close()

    return nwalkers * n_samp * thin_by * n_grid_pts


if __name__ == "__main__":
    # The number of steps taken is thin_param times number_samples
    # As long as number_samples is below 10^5 there should be no RAM issues
    # For larger number_samples, will need to use HDF5 backend of emcee.
    number_samples = 10000
    thin_param = 8
    mcmc_res_file = os.path.join("..", "results", "mcmc", 
                                 "mcmc_results_tcr_tcr_6f.h5")
    if os.path.isfile(mcmc_res_file):
        raise RuntimeError("Existing MCMC results file found at"
            + str(mcmc_res_file) + ". If you want to re-run, "
            + "delete the existing file first. ")
    else:
        print("Starting MCMC runs...")


    # Number of TCR per T cell, L-pulse conversion parameters, peptide taus
    molec_counts_fi = os.path.join("..", "data", "surface_counts", 
                                   "surface_molecule_summary_stats.h5")
    mtc = "Geometric mean"
    nums = load_tcr_tcr_molec_numbers(molec_counts_fi, mtc,
                                        tcell_type="OT1_Naive")
    tcr_number, l_conc_mm_params, pep_tau_map_ot1 = nums

    ## Antagonism ratio fitting
    # Prepare data for fitting antagonism ratios
    data_file_name = os.path.join("..", "data", "antagonism", 
                                  "combined_6f_tcr_tcr_antagonism_datasets.h5")
    df = pd.read_hdf(data_file_name, key="df")
    data_prep = prepare_data_6f(df, l_conc_mm_params, pep_tau_map_ot1)

    # Main run
    nsteps = main_tcr_tcr_run_6f(data_prep, l_conc_mm_params, data_file_name,
                n_samp=number_samples, thin_by=thin_param, R_tot=tcr_number,
                results_file=mcmc_res_file
    )
