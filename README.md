# Theoretical modeling for TCR/CAR interactions

Repository for the code related to mathematical modeling and parameter estimation in the paper

> Taisuke Kondo<sup>=</sup>, François X. P. Bourassa<sup>=</sup>, Sooraj R. Achar<sup>=</sup>, 
J. DuSold, P. F. Céspedes, M. Ando, A. Dwivedi, J. Moraly,
C. Chien, S. Majdoul, A. L. Kenet, M. Wahlsten, A.
Kvalvaag, E. Jenkins, S. P. Kim, C. M. Ade, Z. Yu,
G. Gaud, M. Davila, P. Love, J. C. Yang, M. Dustin,
Grégoire Altan-Bonnet, Paul François, and Naomi Taylor.  __"Engineering TCR-controlled Fuzzy Logic into CAR T-Cells Enhances
Therapeutic Specificity"__,
_Cell_ [accepted in principle], 2025.<br> _(<sup>=</sup>: these authors contributed equally)_

We developed mathematical models to quantitatively predict TCR and CAR cross-receptor interactions in CAR T cells and compare to experimental data. 

The necessary datasets and model results files are hosted on Zenodo:
> [Zenodo repository link](https://doi.org/10.5281/zenodo.14837936)



## Installation

**Step 1**: Download the code with

    git clone https://github.com/frbourassa/tcr_car_antagonism


**Step 2**: __Download the data and code results at:__
<a name="zenodo"> </a>
> [Zenodo repository link](https://doi.org/10.5281/zenodo.14837936).

and unzip the three compressed folders (`results.zip`, `figures.zip`, `data.zip`) , then place them in the main folder of this cloned repository to replace the empty  `results/`, `data/`, and `figures` folders. 

**Step 3**: Install the required packages (listed below). 

The Python scripts and Jupyter notebooks can be run directly, no installation of this repository as a package is required. 


## Requirements

This code was written in Python. It requires typical scientific computing packages (numpy, pandas, scipy, matplotlib, seaborn, etc.) as well as the following uncommon packages:
- [emcee](https://emcee.readthedocs.io/en/stable/)
- [corner](https://corner.readthedocs.io/en/latest/)

It has been tested with Python 3.11.3 and 3.12.7 on Ubuntu 22.04.3 and macOS 14.5 and 15.1.1, with the following package versions:
- corner 2.2.2
- emcee >= 3.1.4  (also tested with emcee 3.1.6)
- matplotlib >= 3.7.1  (also tested with 3.9.2)
- numpy >= 1.24.3 (also tested with numpy 2.0.2)
- pandas >= 1.5.3  (also tested with pandas 2.2.3)
- scipy >= 1.10.1  (also tested with scipy 1.14.1)
- seaborn >= 0.12.2  (also tested with 0.13.2)
- h5py >= 3.7.0  (also 3.11.0)
- scikit-learn >= 1.2.1


## Usage

### Exploring the mathematical models

Numerical evaluation of the mathematical models' solutions is implemented in the `models/` submodules. These submodules are loaded by most other scripts in this repository. 

Additionally, we provide, in the main folder, the notebook `revised_akpr_model_standalone.ipynb`, which is a stand-alone notebook implementing the revised AKPR model for TCR/TCR and TCR/CAR antagonism. It contains all functions to evaluate the steady-state solutions (the same that are available in the `models` module). Parameter values are hardcoded in the notebook, such that it does not depend on data files either. 

Moreover, also in the main folder, we provide the notebook `model_generalized_tcr-car_numbers.ipynb`, where we describe the most general version of the revised AKPR model, for $P$ receptor types each with $q^{\rho}$ ligand types, and provide code to evaluate its analytical steady-state solutions. This model is described in the Supplementary Information and used in figure S6. 

We provide Jupyter notebooks in the folder `notebooks_model_analysis/` to explore in more detail typical TCR/TCR or TCR/CAR model curves. 
- `tcr_tcr_akpr_model_interpretation.ipynb`: illustrates TCR/TCR antagonism in the revised AKPR model
- `tcr_car_akpr_model_interpretation.ipynb`: illustrates TCR/CAR antagonism in the the revised AKPR model. 
- `model_interpretation_6f.ipynb`: similar notebook for 6F TCRs. 
- `manual_tcr_tcr_fitting_tests.ipynb`: to explore parameters in the revised AKPR model for TCR/TCR antagonism and compare to data. 
- `manual_tcr_tcr_fitting_tests_francois2013.ipynb`: to explore parameters in the classical (François et al., 2013) AKPR model and compare to data. 
- `manual_tcr_car_fitting_tests.ipynb`: to explore parameters in the revised AKPR model for TCR/CAR antagonism and compare to data. 
- `tcr_car_models_comparison.ipynb`: compare the original and revised AKPR models for TCR/CAR antagonism. 
- `test_solution_tcr_car_akpr_model.ipynb`: notebook to check that the analytical steady-state solutions of the model match the solutions found by integrating the model ODEs to steady-state.

Except for the stand-alone notebook, datasets and MCMC results should be downloaded (or generated locally) prior to running this code, since it relies on parameter fits and calibrations of ligand numbers and antigen affinities. 


### Predictions of antagonism in peptide libraries

The analysis for antagonism and enhancement predictions in peptide mutant libraries, presented in Figure 5 and explained in Supplemental Information section II.3.9, is performed in two main notebooks, one for each library:

- `universal_antagonism_mskcc_library.ipynb`: for the 7 peptide mutant libraries (3 original peptides, 7 different TCR from patient donors) provided in a recent paper (Luksza *et al*., *Nature*, 2022) by researchers at the Memorial Sloan Kettering Cancer Center (MSKCC). We used our own EC50 fits on their dose response measurements. Some peptides were missing complete dose response curves; we inferred their EC50 based on the response they elicited at maximum dose (that point was measured for all peptides).  
- `universal_antagonism_hhatv4_library.ipynb`: for our HHAT-L8F peptide mutant library, for which we performed our own dose response measurements and EC50 fits. 

The scripts `secondary_scripts/mskcc_ec50_mcmc.py` and `secondary_scripts/hhatv4_ec50_mcmc.py` perform our estimation of peptide EC50s (with uncertainties) by fitting dose response curves with MCMC simulations. We provide the fit results (statistics of the MCMC distributions) in the files `mskcc_ec50_mcmc_stats_backgnd.h5` and `hhatv4_ec50_mcmc_stats_backgnd.h5`, located in the folder `results/pep_libs/` in our [data and results Zenodo repository](#zenodo). Hence, running the MCMC scripts is optional -- note that since there are ~1200 dose response curves to fit, the whole fitting process takes $\sim$ 24 hours on a multi-CPU cluster and generates $\sim 10$ GBs of MCMC samples (not stored on Zenodo). 

Lastly, see `secondary_scripts/universal_antagonism_basic_cell_lines.ipynb` for a similar application, in other cell lines, of the model as a pipeline taking peptide EC50s as an input and returning predicted TCR/CAR fold-changes as an output. 

### Model predictions in various cell types

The core modeling results in Figures 2, 3, 5, and S2 stem from the MCMC scripts and the `universal_antagonism...` notebooks. In addition to these analyses, we used model predictions to motivate experiments in figures 4, 6, S3, S5, and S6. These are mainly applications of the model to various tumor and T cell lines, based on the experimental calibration of antigen and receptor levels on these cell types ([described below](#ligands)). These predictions are generated in the notebook `tcr_car_further_model_predictions.ipynb` for 

- B16-CD19 tumors (Fig. 4);
- SMARTA CAR T cells (CD4+) (Fig. S3);
- Nalm-6 tumors with varying CD19 levels (Fig. S5);
- and AEBS CAR T cells with HHAT antigens (Fig. 6). 

In this latter case, we show that HHAT-WT is an antagonist and HHAT-L8F, an agonist, to motivate our test of the AEBS CAR concept with a Her2 CAR and the HHAT TCR. 

The predictions for mixtures of one CAR antigen and two TCR antigens in Figure S6 are generated by `model_generalized_tcr-car_numbers.ipynb`. 


### Experimental calibration of antigen and receptor parameters
<a name="ligands"> </a>

As reported in figure S2, we calibrated experimentally the number of TCR and CAR antigens and receptors in different cell lines, and we inferred TCR antigen binding times from EC50 measurements. The results of these calibrations are already provided in the [datasets to download](#zenodo), since the model fits and predictions rely on them. The Zenodo page contains a complete list of the provided files; the important end files for users are:

- `data/surface_counts/surface_molecule_summary_stats.h5` provides TCR, CAR, MHC average expression and variability in various cell lines, as well as the Michaelis-Menten curve converting pulse concentrations to a fraction of loaded MHCs. 
- `data/pep_tau_map_ot1.json`, `pep_tau_map_others.json`, `reference_pep_tau_maps.json` provide the model binding times $\tau$ assumed for different TCR antigens, for different TCRs. 
- The dose response fit results for peptide libraries are in `results/pep_libs/hhatv4_ec50_mcmc_stats_backgnd.h5` and `mskcc_ec50_mcmc_stats_backgnd.h5`. 

We also provide the scripts used to generate these calibrations from raw measurements (included in the data repository as well):
- For the EC50 to $\tau$ conversion: `secondary_scripts/ec50_tau_conversion.py`, which relies on the scaling functions defined in `model.conversion`. 
- For the pulse concentration-$L$ calibrations and the receptor and MHC absolute surface levels: `secondary_scripts/surface_molecule_numbers.ipynb` is the Jupyter notebook used to generate `surface_molecule_summary_stats.h5` from raw surface abundance measurements. 
- For fitting dose response curves for OT-1, NY-ESO, and HHAT TCR peptides: `secondary_scripts/fitting_EC50s_dose_response.ipynb`. This generates `data/dose_response/experimental_peptide_ec50s_blasts.h5`, as well as the OT-1 cytokine dose response curves in Fig. S1. 
  - Note that these cytokine- or CD25-based EC50s for OT-1 peptides are not used in the model, we relied on previous estimates from Achar *et al*., 2022 in `potencies_df_2021.json`; the experiments in Fig. S1 were done to confirm that regular OT-1 T cells have similar TCR antigen dose responses compared to 1- and 3-ITAM OT-1 CAR T cells. 


### Running MCMC parameter estimation simulations

Original MCMC samples are provided in the data repository, so this part is optional. The results can be reproduced by running the scripts in the `run_mcmc_scripts/` folder. `cd` to this folder, then run scripts in the following order. 
1. `run_mcmc_tcr_tcr_akpr_i.py` and `run_mcmc_tcr_tcr_shp1.py` (can be run in parallel or in either order). 
2. `run_mcmc_tcr_tcr_6f.py`
3. `analyze_mcmc_tcr_tcr.py`
4. `run_mcmc_tcr_car.py`
5. `analyze_mcmc_tcr_car.py`

The order matters, as TCR/TCR model fits need to be generated and analyzed before TCR/CAR simulations are launched. These MCMC scripts should take in total 12-24h to run on a desktop computer or small computational cluster with multiple CPU cores. You can launch test runs first by lowering the simulation length, `n_steps`, in the `__main__` of these scripts. 

The script `run_mcmc_withpriors.py` runs all MCMC simulations for an alternate choice of parameter priors $P(\theta)$ -- log-normal distributions centered on the original AKPR model parameters, with variances picked based on the previous simulations. These alternate simulations were not used in the paper but were run to show that better solutions have not been ignored by the biologically plausible parameter boundaries we picked.  The Zenodo repository contains figures and analysis results produced by these simulations (but not the full MCMC chains, since these are large files). 

The model fits and predictions of antagonism data in OT-1 T cells are part of the MCMC simulation results. The scripts `analyze_mcmc_tcr_tcr.py` (Figure 2) and `analyze_mcmc_tcr_car.py` (Figure 3) generate model fits and predictions plots for each $(k, m, f)$ over which the grid search is performed. The final figures in the paper are for the $(k, m, f)$ providing the best fit. 
- For TCR/TCR fits and parameter distributions (Figures 2 and S2): `figures/mcmc_shp1` (classical model), `figures/mcmc_akpr_i` (revised AKPR model),  `figures/mcmc_tcr_tcr_6f/` (revised AKPR for 6F T cells);
- For TCR/CAR fits and parameter distributions (Figure 3): `figures/mcmc_tcr_car/`;
- For TCR/CAR predictions in various conditions (Figure 3): `figures/model_predictions/`. 
See a comparison of all $(k, m, f)$ grid points for TCR/TCR antagonism in terms of the Akaike information criterion in `figures/model_comparison/models_comparison_aic_mcmc_fits.pdf`. 


### Correlation between model and *in vivo* results

The Jupyter notebook `secondary_scripts/tcr_car_invivo_predictions.ipynb` shows that the model predictions of TCR/CAR antagonism, fitted on *in vitro* data, correlate relatively well with mouse *in vivo* survival data for tumors with OT-1 antigens (Figure 4, file `data/invivo/B16-combined_tumorMeasurements-unblinded.h5`). Performing a linear regression of the model output versus survival in single antigen tumors (TCR or CAR only), we attempted to predict survival in dual antigen tumors (TCR and CAR). There are, however, outliers, so extrapolation of the model, as it currently stands, to *in vivo* or clinical settings is not warranted. 


### Reproducing final figures from the paper

Jupyter notebooks generating published figure panels related to the model are provided in the `plotting_final/` folder. This code is essentially taking care of aesthetic plotting details; no new analyses are performed therein. It relies on some plotting scripts in `plotting_final/scripts/` and on various data and output files which are part of the [data and results Zenodo repository](#zenodo). The panels are saved in subfolders within `plotting_final/`. We did not include these final plots in the Zenodo repository because, well, they are in the paper. 

Two scripts in `secondary_scripts`, `mcmc_ci_to_latex.py` and `print_dataset_sizes.py`, generate text outputs used for supplementary tables. 

## Authors and acknowledgments

We acknowledge the full list of authors who contributed to the work presented in the paper and collected the experimental data (see above). 

Main author of the theory-related code: **François X. P. Bourassa** (https://orcid.org/0000-0002-2757-5870), with important contributions from **Sooraj R. Achar** (https://orcid.org/0000-0003-3425-7501) in data analysis and plotting scripts, and with supervision from **Naomi Taylor**, **Grégoire Altan-Bonnet**, and **Paul François**. 
**Taisuke Kondo**, **Sooraj Achar**, and other authors performed the experiments on which the theoretical analyses relied. 


## License
Code: BSD-3-Clause

Data on the separate Zenodo repository: CC-BY-4.0

## Badges
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14837936.svg)](https://doi.org/10.5281/zenodo.14837936)