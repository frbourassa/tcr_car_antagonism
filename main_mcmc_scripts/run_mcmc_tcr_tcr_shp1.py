"""
Fit antagonism ratio data. Grid search integer parameters $m$,
and for each m, optimize feedback parameters $C_{m, thresh}$ and $S_T$.

Keeping separate main script for each model so we can launch MCMC in parallel
on different clusters.

@author: frbourassa
November 2022
"""

import numpy as np
import pandas as pd
import scipy as sp
from scipy import optimize
import h5py, json
from time import perf_counter

# Local modules
import sys, os
if not "../" in sys.path:
    sys.path.insert(1, "../")

from models.shp1_model import steady_shp1_1ligand, steady_shp1_2ligands
from mcmc.costs_tcr_tcr_antagonism import cost_antagonism_shp1
from mcmc.mcmc_run import grid_search_emcee
from mcmc.utilities_tcr_tcr_antagonism import (
    prepare_data,
    load_tcr_tcr_molec_numbers
)
from utils.preprocess import loglog_michaelis_menten, time_dd_hh_mm_ss

def main_shp1_run(data_list, mm_params, data_fname, n_samp, thin_by, R_tot):
    df_fit, df_ci_log2, tau_agonist = data_list
    # Define the parameters that will remain the same throughout
    # Parameters related to the cascade of phosphorylations
    phi = 0.09
    b = 0.04
    gamma = 1.2e-6
    kappa = 1e-4

    # Number of receptors: based on calibration data, passed as arg R_tot

    # Number of steps, etc.
    N = 5

    # Initial values of parameters to fit. The best I found manually
    # Gamma and s_tot compensate each other; just fit S_tot
    cm_thresh = 500
    S_tot = 6e5
    m_feed = 2

    # Bounds on parameters: phi, cmthresh, s_tot
    fit_bounds = [(0.1, 5.0), (1.0, 100*R_tot), (S_tot/1000, 1000*S_tot)]
    fit_bounds = [np.log10(np.asarray(a)) for a in zip(*fit_bounds)]

    # Wrapping up parameters
    rates_others = [b, gamma, kappa]
    n_m = [N, m_feed]  # m will be gridded over
    fit_params_vec = np.log10(np.asarray([phi, cm_thresh, S_tot]))

    # Fit
    # Grid search over m, f, k_S, each time using simulated annealing
    # to adjust C_m threshold, S threshold, and psi_0
    # Note: this can take a long time; run on the physics cluster
    m_bounds = [(1, 5)]
    n_grid_pts = np.product([a[1]-a[0]+1 for a in m_bounds])
    nwalkers = 32

    seed_sequence = np.random.SeedSequence(0x4a31b78fd6f910991979ed8988feb84a,
                            spawn_key=(0xa30cf5c0f1727ac7805c077c00af3d0f,))
    # pvec, pbounds, mp, other_rates, rtot, n_p, ag_tau, df_ratio, df_err
    cost_args = (rates_others, R_tot, N, tau_agonist, df_fit, df_ci_log2)
    # List only the names of cost function args to save to the hdf5 file.
    cost_args_names = ["rates_others", "R_tot", "N", "tau_agonist"]
    results_file = "../results/mcmc/mcmc_results_shp1_test.h5"
    try:
        start_t = perf_counter()
        grid_search_emcee(cost_antagonism_shp1,
            m_bounds, fit_bounds, results_file, p0=None, nwalkers=nwalkers,
            nsamples=n_samp, seed_sequence=seed_sequence, cost_args=cost_args,
            cost_kwargs={}, emcee_kwargs={}, run_kwargs={"tune":True, "thin_by":thin_by},
            param_names=[r"$\varphi^T$", r"$C^T_{m,th}$", r"$I^T_{tot}$"])
    except Exception as e:
        os.remove(results_file)
        raise e
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
        args_group.attrs["data_file_name"] = data_fname
        args_group.attrs["thin_by"] = thin_by
        args_group.attrs["run_time"] = delta_t
        args_group.attrs["nsteps"] = nsteps
        for i in range(len(cost_args_names)):
            args_group.create_dataset(cost_args_names[i], data=cost_args[i])
        # Add the L-pulse conversion parameters too.
        args_group.create_dataset("l_conc_mm_params", data=l_conc_mm_params)
        results_obj.close()

    return nsteps


if __name__ == "__main__":
    # The number of steps taken is thin_param times number_samples
    # As long as number_samples is below 10^5 there should be no RAM issues
    # For larger number_samples, will need to use HDF5 backend of emcee.
    number_samples = 10000
    thin_param = 4

    # Number of TCR per T cell, L-pulse conversion parameters, peptide taus
    molec_counts_fi = "../data/surface_counts/surface_molecule_summary_stats.h5"
    mtc = "Geometric mean"
    nums = load_tcr_tcr_molec_numbers(molec_counts_fi, mtc,
                                        tcell_type="OT1_Naive")
    tcr_number, l_conc_mm_params, pep_tau_map_ot1 = nums

    ## Antagonism ratio fitting
    # Prepare data for fitting antagonism ratios
    data_file_name = "../data/antagonism/allManualSameCellAntagonismDfs_v3.h5"
    df = pd.read_hdf(data_file_name)
    data_prep = prepare_data(df, l_conc_mm_params, pep_tau_map_ot1)

    # Main run
    nsteps = main_shp1_run(data_prep, l_conc_mm_params, data_file_name,
                number_samples, thin_param, tcr_number)
