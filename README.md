# Theoretical modelling for TCR/CAR interactions

Repository for the code related to mathematical modelling and parameter estimation in the paper

> Taisuke Kondo<sup>=</sup>, François X. P. Bourassa<sup>=</sup>, Sooraj R. Achar<sup>=</sup>, J.
DuSold, P. F. Céspedes, M. Ando, A. Dwivedi, J. Moraly,
C. Chien, S. Majdoul, A. L. Kenet, M. Wahlsten, A.
Kvalvaag, E. Jenkins, S. P. Kim, C. M. Ade, Z. Yu,
G. Gaud, M. Davila, P. Love, J. C. Yang, M. Dustin,
Grégoire Altan-Bonnet, Paul François, and Naomi Taylor.  __"Engineering TCR-controlled Fuzzy Logic into CAR T-Cells Enhances
Therapeutic Specificity"__,
_Cell_ [accepted in principle], 2025.<br> _(<sup>=</sup>: these authors contributed equally)_

in which we developed mathematical models to quantitatively predict TCR and CAR cross-receptor interactions in CAR T cells and compare to experimental data. 

The necessary datasets and model results files are hosted on Zenodo:
> [Zenodo repository link](https://zenodo.org/)



## Installation
__Download the data and MCMC results at:__
> [Zenodo repository link](https://zenodo.org/).

and place them in the main folder of this repository to replace the  `results/` and `data/` folders. 

Then, install the required packages (listed below) and download the code. The Python scripts and Jupyter notebooks can be run directly, no installation required. 

## Requirements

This code requires typical scientific computing packages (numpy, pandas, scipy, matplotlib, seaborn, etc.) as well as the following uncommon packages:
- [emcee](https://emcee.readthedocs.io/en/stable/)
- [corner](https://corner.readthedocs.io/en/latest/)

It has been tested with Python 3.11.3 and 3.12.7 on Ubuntu 22.04.3 and macOS 14 and 15.1.1, with the following package versions:
- corner 2.2.2
- emcee >= 3.1.4  (also tested with emcee 3.1.6)
- matplotlib >= 3.7.1  (also tested with 3.9.2)
- numpy >= 1.24.3 (also tested with numpy 2.0.2)
- pandas >= 1.5.3  (also tested with pandas 2.2.3)
- scipy >= 1.10.1  (also tested with scipy 1.14.1)
- seaborn >= 0.12.2  (also tested with 0.13.2)
- h5py >= 3.7.0  (also 3.11.0)
- statsmodels >= 0.13.5
- sympy >= 1.11.1
- scikit-learn >= 1.2.1


## Usage

### Exploring the mathematical models
Use the following Jupyter notebooks to explore typical TCR/TCR or TCR/CAR model curves. 
- `tcr_tcr_akpr_model_interpretation.ipynb`: illustrates TCR/TCR antagonism in the revised AKPR model
- `secondary_scripts/manual_tcr_tcr_fitting_tests.ipynb`: to explore parameters in the revised AKPR model for TCR/TCR antagonism and compare to data. 
- `secondary_scripts/manual_tcr_tcr_fitting_tests_francois2013.ipynb`: to explore parameters in the classical (François et al., 2013) AKPR model and compare to data. 
- `tcr_car_akpr_model_interpretation.ipynb`: illustrates TCR/CAR antagonism in the the revised AKPR model. 
- `secondary_scripts/manual_tcr_car_fitting_tests.ipynb`: to explore parameters in the revised AKPR model for TCR/CAR antagonism and compare to data. 
- `secondary_scripts/tcr_car_model_comparison.ipynb`: 

Of note, we provide the notebook `secondary_scripts/revised_akpr_model_standalone.ipynb`, which is a stand-alone notebook implementing the model: it contains all functions to evaluate the TCR/TCR and TCR/CAR model steady-state solutions, without importing from other modules in this repository. Parameter values are hardcoded in the notebook, such that it does not depend on data files either. 

Moreover, in `model_generalized_tcr-car_numbers.ipynb`, we describe the most general version of the revised AKPR model, for $P$ receptor types each with $q^{\rho}$ ligand types, and provide code to evaluate its analytical steady-state solutions. This model is described in the Supplementary Information. 

Datasets and results should be downloaded (or generated locally) prior to running these notebooks, since the best parameter fits, the ligand numbers and antigen affinities are loaded in them. 

### Running MCMC parameter estimation simulations
Original MCMC samples are provided in the data repository, so this part is optional. The results can be reproduced by running the scripts in the `optimization_mcmc/` folder. `cd` to this folder, then run scripts in the following order. 
1. `run_mcmc_tcr_tcr_akpr_i.py` (and `run_mcmc_tcr_tcr_shp1.py` can be run in parallel to compare to that model). 
2. `run_mcmc_tcr_tcr_6f.py`
3. `analyze_mcmc_tcr_tcr.py`
4. `run_mcmc_tcr_car.py`
5. `analyze_mcmc_tcr_car.py`

The order matters, as TCR/TCR model fits need to be generated and analyzed before TCR/CAR simulations are launched. These simulations should take in total 12-24h to run on a desktop computer or small computational cluster with multiple CPU cores. Seeds for the random generators can be changed in the `main_...` functions. 

The script `run_mcmc_withpriors.py` runs all MCMC simulations in a similar order for an alternate choice of parameter priors $P(\theta)$ -- log-normal distributions centered on the original AKPR model parameters, with variances picked based on the previous simulations. These simulations were not used in the main paper but were run to show that better solutions have not been ignored by the parameter boundaries of the default simulations above. 



### Predictions of antagonism in peptide libraries
The analysis for antagonism and enhancement predictions in peptide mutant libraries, presented in Figure 5, is performed in two main notebooks for the two libraries:
- `

Revised peptide EC50 estimation procedure from the dose response data in the MSKCC paper (Luksza *et al*., *Nature*, 2022). All weaker CMV-derived antigens did not have full (3 concentration points) dose response curves and thus infinite EC50s. This would have led to a systematic under-estimation of the fraction of antagonists in these peptide sets (since the weaker peptides that were missing dose responses are those most likely to antagonize). 

    
Figure 5: see `universal_antagonism_mskcc_library.ipynb` and `universal_antagonism_hhatv4_library.ipynb` notebooks for the model FC predictions in MSKCC and HHAT peptide libraries. See our peptide EC50 fits (especially the revised ones for the MSKCC dataset) in `secondary_scripts/mskcc_ec50_mcmc.py` and `secondary_scripts/hhatv4_ec50_mcmc.py`. See also `secondary_scripts/universal_antagonism_basic_cell_lines.ipynb`. 


### Model predictions in various cell types
In addition to the core modelling results in Figures 2, 3, 5, and S2, we used model predictions to motivate experiments in figures 4, 6, S3, and S5 contain model predictions for other tumor lines than E2aPBX and other CAR T cell types than OT-1, to motivate the experiments carried out in each case. Also, for figure 5, the model is applied to peptide libraries: see the next subsection. 

- Figure 4: see `tcr_car_invivo_predictions.ipynb` for predictions in B16 tumors (high CD19, low MHC). 

- Figures S3, S5, 6: see `tcr_car_further_model_predictions.ipynb` for SMARTA CAR T cells (CD4+), Nalm-6 tumors with varying CD19 levels, and AEBS CAR T cells with HHAT antigens. Show that HHAT-WT is an antagonist and HHAT-L8F, an agonist, based on their dose response curves and EC50s.  



### Experimental calibration of antigen and receptor parameters
The results of these calibrations are already provided in the downloaded datasets, since the model fits and predictions rely on them:
- `data/surface_counts/surface_molecule_summary_stats.h5` provides TCR, CAR, MHC average expression and variability in various cell lines, as well as the Michaelis-Menten curve converting pulse concentrations to a fraction of loaded MHCs. 
- `data/pep_tau_map_ot1.json`, `pep_tau_map_others.json`, `reference_pep_tau_maps.json` provide the model binding times $\tau$ assumed for different TCR antigens, for different TCRs. 
- There are raw dose response curve measurements (not converted to $\tau$s) in the `data/dose_response/` folder. 
- For the MSKCC peptide libraries and our

We also provide the scripts used to generate these calibrations:
- For the EC50 to $\tau$ conversion: `secondary_scripts/ec50_tau_conversion.py`, which relies on the scaling functions defined in `model.conversion`. 
- For the pulse concentration-$L$ calibrations and the receptor and MHC absolute surface levels: `secondary_scripts/surface_molecule_numbers.ipynb`  is the Jupyter notebook used to generate `surface_molecule_summary_stats.h5` from raw surface abundance measurements in `data/surface_counts/` (for molecule numbers: `completeMoleculeNumberDf.hdf`; for MHC loading: `combinedRMASDf_MFI.hdf`). 
- For fitting dose response curves for OT-1, NY-ESO, and HHAT TCR peptides: `secondary_scripts/OT_1_EC50s.ipynb`. This generates `data/dose_response/experimental_peptide_ec50s_blasts.h5`, as well as the OT-1 cytokine dose response curves in Fig. S1. 
  - Note that these OT-1 cytokine-based EC50s are not used in the model, we relied on previous estimates from Achar *et al*., 2022 in `potencies_df_2021.json`; the experiments in Fig. S1 were done to confirm that regular OT-1 T cells have similar TCR antigen dose responses compared to 1- and 3-ITAM OT-1 CAR T cells. 


### Correlation between model and *in vivo* results
The Jupyter notebook `secondary_scripts/tcr_car_invivo_predictions.ipynb` shows that the model predictions of TCR/CAR antagonism, fitted on *in vitro* data, correlate relatively well with mouse *in vivo* survival data for tumors with OT-1 antigens (Figure 4). Performing a linear regression of the model output versus survival in single antigen tumors (TCR or CAR only), we attempted to predict survival in dual antigen tumors (TCR and CAR). There are, however, outliers, and the poor quantitative agreement does not support confident extrapolation of the model to *in vivo* or clinical settings. 



### Reproducing figures from the paper
TODO: include figure code in a folder? Not sure I will have time. 

The model fits and predictions of antagonism data in OT-1 T cells are part of the MCMC simulation results. Run the MCMC analysis scripts to generate model fit figures: `analyze_mcmc_tcr_tcr.py` (Figure 2) and `analyze_mcmc_tcr_car.py` (Figure 3). The following figures will be generated and saved for each $(k, m, f)$ over which the grid search is performed. 
- For TCR/TCR fits and parameter distributions (Figures 2 and S2): `figures/mcmc_shp1` (classical model), `figures/mcmc_akpr_i` (revised AKPR model),  `figures/mcmc_tcr_tcr_6f/` (revised AKPR for 6F T cells);
- For TCR/CAR fits and parameter distributions (Figure 3): `figures/mcmc_tcr_car/`;
- For TCR/CAR predictions in various conditions (Figure 3): `figures/model_predictions/`. 



## Authors and acknowledgments

We acknowledge the full list of authors provided above who contributed to the manuscript. 

Main author of the theory-related code: **François X. P. Bourassa** (https://orcid.org/0000-0002-2757-5870), with contributions from **Sooraj R. Achar** in data analysis and plotting scripts, and with supervision from **Paul François** and **Grégoire Altan-Bonnet**. 


## License
BSD-3-Clause

## Badges
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5758439.svg)](https://doi.org/10.5281/zenodo.5758439)