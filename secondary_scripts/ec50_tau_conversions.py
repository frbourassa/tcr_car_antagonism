"""
# Experimental EC50 to model binding time (tau) conversion

Method: we assume $\tau$ and the EC50 $L_{50}$ (or EC10, etc.) are related by an
ideal KPR scaling law, independent of the specific model chosen:

$$ L_{50} \tau^N = \Theta $$

where $\Theta$ is some threshold, $L_{50}$ is the EC50 in ligand numbers,
and $\tau$ is the binding time corresponding to that EC50.

Moreover, we assume that ligand numbers scale proportionally with
pulse concentration. This is entirely true for stronger antigens with $L_{50}$
below the number of MHC per APC. For weak antigens, we neglect APC saturation
since anyways their EC50 is an idealization, the number of ligands that would
produce half-max. response if we could go that far.
Hence, up to a rescaling of the threshold $\Theta$, we can replace L in the
equation above by the experimental EC50 $P_{50}$ expressed in pulse
concentration units:

$$ P_{50} \tau^N = \Theta' $$

Now, if we have or choose a reference $\tau_{ref}$ for a peptide with known
EC50 $P_{ref}$, we can determine the threshold to be

$$ \Theta' = P_{ref} \tau_{ref}^N $$

and hence, every other peptide's $\tau$ is determined as

$$ \tau = \tau_{ref} (P_{ref} / P_{50})^{1/N}  $$

or, if we know the EC50 of the peptide relative to the reference,

$$ \tau = \tau_{ref} (EC_{50, rel})^{-1/N} $$

We use the last two formulas to determine taus for every peptide involved
in the experiments of this project. For OT-1 peptides, we have EC50s relative
to N4 measured with naive T cells, from Daniels 2006, Zehn 2009, and our
own assays (Achar 2022). We choose as a reference $\tau = 10$ s for the
OVA antigen N4 (SIINFEKL).

For other peptides, we have dose response data based on % CD25 + T cells
for OT-1 peptides and for other peptides. This allows us to compare the EC50
of all peptides in pulse concentration units, to N4, which we still set
at $\tau = 10$ s.

For very weak peptides that produce basically no response in dose response
experiments, antagonism is a better metric for their potency. The best approach
is therefore to estimate their value by fitting antagonism data.

 - OT-1 system: we fit the binding time of E1 (EIINFEKL1) along
with the other model parameters fitted on TCR-TCR antagonism data.

 - NY-ESO: see later.
Other peptides we used produce some EC50 response.


@author: frbourassa
April 2023
"""
import numpy as np
import json
import pandas as pd
import os

import sys
if "../" not in sys.path:
    sys.path.insert(1, "../")
from models.conversion import convert_ec50_tau_thresh, convert_ec50_tau_relative
from utils.export import nice_dict_print, save_defaults_json


# OT-1 peptides
def find_tau_ot1_peptides(ec50_file, ref_file, ot1_file,
                        ref_pep="N4", overwrite=False):
    # Import relative EC50s determined with naive cells/from literature
    ec50data = pd.read_json(ec50_file)
    # Geometric average
    ec50data = np.exp(np.log(ec50data).mean(axis=1))

    # Read off reference point from a file
    with open(ref_file, "r") as file:
        tau_refs = json.load(file)
    npow = tau_refs["npow"]

    # Convert relative EC50s to tau
    converter = lambda x: convert_ec50_tau_relative(x, tau_refs[ref_pep], npow)
    tau_map = ec50data.map(converter).to_dict()

    # Add in None
    tau_map["None"] = tau_refs["None"]

    # Save to JSON, not replacing existing values. E1 to be updated later
    tau_map = save_defaults_json(tau_map, ot1_file, overwrite=overwrite)
    return tau_map

# Other peptides, compared to OT-1, based on CD25 for comparison
# Alternative reference point: use absolute pulse concentrations from some
# other modality for OT-1, then convert CD50 EC50 pulse of other peptides.
def find_tau_other_peptides(ec50_file, ref_file, ot1_file, others_file,
                        ref_pep="OT1-N4", overwrite=False):
    # Import experimental EC50s determined with blasts across T cell types
    ec50data = pd.read_hdf(ec50_file)
    # Create a new index level combining TCR and peptide
    new_lvl = (ec50data.index.get_level_values("TCR") + "-"
                                + ec50data.index.get_level_values("Peptide"))
    ec50data.index = pd.MultiIndex.from_tuples([(new_lvl[i], *a)
                        for i, a in enumerate(ec50data.index)],
                        names=["TCR-Peptide"] + list(ec50data.index.names))
    # Geometric average
    ec50data = np.exp(np.log(ec50data).xs("CD25", level="Marker")
                .groupby("TCR-Peptide").mean()) * 1e6  # in uM
    # Compute EC50s relative to reference point
    ec50data_rel = ec50data / ec50data.loc[ref_pep]

    # Read off reference point from a file
    with open(ot1_file, "r") as file:
        ot1_refs = json.load(file)
    with open(ref_file, "r") as file:
        tau_refs = json.load(file)
    try:
        tau_ref = tau_refs[ref_pep.split("-")[1]]
    except KeyError:
        tau_ref = ot1_refs[ref_pep.split("-")[1]]
    npow = tau_refs["npow"]

    # Drop OT-1 peptides, already determined
    ot1_refs.pop(ref_pep.split("-")[1])
    ec50data_rel = ec50data_rel.drop(["OT1-"+ a for a in ot1_refs.keys()],
                                    errors="ignore")

    # Convert relative EC50s to tau
    converter = lambda x: convert_ec50_tau_relative(x, tau_ref, npow)
    tau_map = ec50data_rel.map(converter).to_dict()

    # Add in None
    tau_map["None"] = tau_refs["None"]

    # Save to JSON, not replacing existing values.
    tau_map = save_defaults_json(tau_map, others_file, overwrite=overwrite)
    return tau_map


if __name__ == "__main__":
    # OT-1 peptides
    pep_tau_map_ot1 = find_tau_ot1_peptides(
        ec50_file="../data/dose_response/potencies_df_2021.json",
        ref_file="../data/reference_pep_tau_maps.json",
        ot1_file="../data/pep_tau_map_ot1.json",
        ref_pep="N4",
        overwrite=False
    )

    print("Current OT-1 peptides map:")
    nice_dict_print(pep_tau_map_ot1)
    print("To overwrite existing file, re-run with overwrite=True")

    # Other peptides
    pep_tau_map_others = find_tau_other_peptides(
        ec50_file="../data/dose_response/experimental_peptide_ec50s_blasts.h5",
        ref_file="../data/reference_pep_tau_maps.json",
        ot1_file="../data/pep_tau_map_ot1.json",
        others_file="../data/pep_tau_map_others.json",
        ref_pep="OT1-N4",
        overwrite=True
    )

    print("Current other peptides map:")
    nice_dict_print(pep_tau_map_others)
