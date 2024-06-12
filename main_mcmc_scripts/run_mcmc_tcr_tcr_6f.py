"""
Fit recent TCR-TCR antagonism ratio data for 6F T cells
(mutated Y->F on the six CD3z ITAMs). These cells have different
KPR and inhibitory module activation properties (Gaud et al., under review)
so the model parameters have to be re-estimated separately for them (instead
of enforcing parameter modification by hand, use MCMC for unbiased fit).

Setting N=4 for these cells, they have poorer ligand discrimination abilities
and fewer ITAM phosphorylation steps at the beginning of the cascade.
Expecting m=1 is best (seeing that E1 is the best antagonist for them).

Only using the updated AKPR model (inhibitory module, nonlinear psi)
since it was shown to be better than the SHP-1 model already.

Grid search integer parameters $m$, $f$, $k_S$
and for each, optimize $C_{m, thresh}$, $S_0$, and $\psi_0$.
It would be fair to change m, f, k_S but if many choices are very similar,
prioritize the same choices as for 6Y TCRs (except maybe m).

@author: frbourassa
March 2023
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

from models.akpr_i_model import steady_akpr_i_1ligand, steady_akpr_i_2ligands
from mcmc.costs_tcr_tcr_antagonism import cost_antagonism_akpr_i
from mcmc.mcmc_run import grid_search_emcee
from mcmc.utilities_tcr_tcr_antagonism import (
    prepare_data_6f,
    load_tcr_tcr_molec_numbers
)
from utils.preprocess import loglog_michaelis_menten, time_dd_hh_mm_ss


def main_tcr_tcr_run_6f(data_list, mm_params, data_fname, n_samp, thin_by, R_tot):
    df_fit, df_ci_log2, tau_agonist = data_list
    # Model parameters
    # Define the parameters that will remain the same throughout
    # Parameters related to the cascade of phosphorylations
    phi = 0.2
    kappa = 1e-4
    psi_0 = 0.01

    # Normalized SHP-1 amount
    S_tot = 1.0

    # Number of TCR: based on calibration data, passed as argument R_tot

    # Number of steps, etc.
    N = 4

    # Initial values of parameters to grid over.
    k_S = 1
    m_feed = 2
    f_last = 2

    # Initial values for parameters to fit. Different from manual results.
    S_thresh = 0.5
    cm_thresh = 300
    # Based on Gaud et al., 2023, we know that a good way to explain 6F vs 6Y
    # is to increase the activation threshold of I by 6F compared to 6Y.
    # In other words, 6F has poorer activation of the negative feedback.
    # Here in this model, the product of C_mthresh and I_thresh is what really
    # sets the threshold of inhibitory influence. By default, MCMC
    # finds that 6F has that product 4x larger than 6Y, meaning less inhibition,
    # but it finds a low C_mthresh and a high I_thresh, which has a different
    # biochemical interpretation -- it would mean easy trigger of I, but
    # I has little effect. This is entirely possible as E1 antagonizes the
    # CAR more than in 6Y TCR CAR T cells, indicating strong but quickly
    # saturating inhibitory coupling. So we may need to revise the initial
    # interpretation: 6F is less susceptible to SHP-1 inhibition, but it
    # activates it just as much.
    fit_param_names = [r"$\varphi^T$", r"$C^T_{m,th}$",
                    r"$I^T_{th}$", r"$\psi^T_0$"]
    # Bounds on parameters: phi, cmthresh, sthresh, psi_0
    # Use relatively tight bounds based on previous runs.
    # Use the log of parameters so the MCMC steps are even in log scale
    fit_bounds = [(0.05, 5.0), (1.0, 100*R_tot), (1e-5, 1000*S_tot), (1e-8, 1e-1)]
    fit_bounds = [np.log10(np.asarray(a)) for a in zip(*fit_bounds)]

    # Wrapping up parameters
    rates_others = [kappa]  # k_S will be gridded over
    n_m_f = [N, m_feed, f_last]  # m and f will be gridded over
    total_RI = [R_tot, S_tot]
    fit_params_vec = np.log10(np.asarray([cm_thresh, S_thresh]))

    # Fit
    # Grid search over m, f, k_S, each time using simulated annealing
    # to adjust C_m threshold, S threshold, and psi_0
    # Do not include k=2 as we know this doesn't work well on 6Y,
    # the 6F is even simpler so k=2 would be hard to motivate,
    # and it seems to lead to overfitting when making TCR/CAR predictions
    kmf_bounds = [(1, 1), (1, 4), (1, 2)]
    n_grid_pts = np.product([a[1]-a[0]+1 for a in kmf_bounds])
    nwalkers = 32

    seed_sequence = np.random.SeedSequence(0x92ea74f33e7a98edefc0c171aee692f1,
                            spawn_key=(0xc2b221db89ad05809ce1125db6515652,))
    # Skip computing MI in the cost function
    cost_args = (rates_others, total_RI, N, tau_agonist, df_fit, df_ci_log2)
    # List only the names of cost function args to save to the hdf5 file.
    cost_args_names = ["rates_others", "total_RI", "N", "tau_agonist"]
    results_file = "../results/mcmc/mcmc_results_tcr_tcr_6f_test.h5"
    try:
        start_t = perf_counter()
        grid_search_emcee(cost_antagonism_akpr_i,
            kmf_bounds, fit_bounds, results_file, p0=None, nwalkers=nwalkers,
            nsamples=n_samp, seed_sequence=seed_sequence, cost_args=cost_args,
            cost_kwargs={}, emcee_kwargs={}, run_kwargs={"tune":True, "thin_by":thin_by},
            param_names=fit_param_names)
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
        args_group.attrs["data_file_name"] = data_fname
        args_group.attrs["thin_by"] = thin_by
        args_group.attrs["run_time"] = delta_t
        args_group.attrs["nsteps"] = nsteps
        for i in range(len(cost_args_names)):
            args_group.create_dataset(cost_args_names[i], data=cost_args[i])
        # Add the L-pulse conversion parameters too.
        args_group.create_dataset("l_conc_mm_params", data=l_conc_mm_params)
        results_obj.close()

    return nwalkers * n_samp * thin_by * n_grid_pts


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
    data_file_name = "../data/antagonism/combined_6f_tcr_tcr_antagonism_datasets.h5"
    df = pd.read_hdf(data_file_name, key="df")
    data_prep = prepare_data_6f(df, l_conc_mm_params, pep_tau_map_ot1)

    # Main run
    nsteps = main_tcr_tcr_run_6f(data_prep, l_conc_mm_params, data_file_name,
                    number_samples, thin_param, tcr_number)
