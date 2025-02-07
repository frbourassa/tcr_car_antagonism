""" Script to automatically collect the CIs on best fits from MCMC simulations,
round them to the appropriate number of significant digits,
and .

Report log10 values of parameters in the format:

best fit    (- to 5th percentile)   (+ to 95th percentile)
"""
import json, os, sys, h5py
import numpy as np
import pandas as pd
from math import floor, log10

if not "../" in sys.path:
    sys.path.insert(1, "../")
from mcmc.mcmc_analysis import find_best_grid_point

def percentile_symbol(p, place=-1):
    r""" Given a percentile p, generate a math tex string with a shorthand
    symbol for percentiles. Currently using format $P_{{p}\%}$ """
    # Round p as indicated
    if place > 0:
        pr = round(p, place)
        s = "$P_{" + f"{pr:.{place}f}" + r"\%}$"
    else:
        pr = int(p)
        s = "$P_{" + str(pr) + r"\%}$"
    return s


def ci_symbol(p, place=-1):
    r""" Given a percentile p, generate a math tex string with a shorthand
    symbol for CI limits. Currently using format $CI_{{p}\%}$ """
    # Round p as indicated
    if place > 0:
        pr = round(p, place)
        s = r"$\mathrm{CI}_{" + f"{pr:.{place}f}" + r"\%}$"
    else:
        pr = int(p)
        s = r"$\mathrm{CI}_{" + str(pr) + r"\%}$"
    return s


def get_best_cis_names(res_file, best_cond=None):
    """ Store best, lower, upper in each row of an array """
    with open(res_file, "r") as f:
        res_json = json.load(f)
    if best_cond is None:
        bests = find_best_grid_point(res_json, strat="best")
        best_cond, best_p, _ = bests
    else:
        best_p = res_json.get(best_cond).get("param_estimates").get("MAP best")
        best_p = np.asarray(best_p)
    ci_dict = res_json.get(best_cond).get("confidence_intervals")
    pnames = list(ci_dict.keys())
    best_and_cis = np.zeros([len(pnames), 3])
    # Put best in the middle
    best_and_cis[:, 1] = best_p
    for i, p in enumerate(pnames):
        # Put CIs around
        best_and_cis[i, 0::2] = ci_dict.get(p)
    # Create a dataframe
    pnames = [r"$\log_{10} " + p[1:] for p in pnames]
    return best_and_cis, pnames


def round_best_cis(pv):
    """ For each row in pv giving (best, lower CI, upper CI),
    """
    # Find the differences between best and CIs to determine significant digits
    diffs = np.hstack([pv[:, 1:2] - pv[:, 0:1], pv[:, 2:3] - pv[:, 1:2]])
    place10 = np.amin(np.floor(np.log10(np.abs(diffs))), axis=1)
    digits = -place10.astype(int)
    pv_round = np.zeros(pv.shape)
    for i in range(pv.shape[0]):
        pv_round[i] = np.around(pv[i], digits[i])
    return pv_round, digits


def round_to_place(x, f):
    """ Round x to the fth power of 10 """
    return 10**f * np.around(x / 10**f, 0)


def round_best_cis_scinot(pv):
    """ Round best estimate and CIs to the appropriate number of significant
    digits in scientific notation, based on the difference between best
    and closest CI.
    """
    diffs = np.hstack([pv[:, 1:2] - pv[:, 0:1], pv[:, 2:3] - pv[:, 1:2]])
    # Power of 10 of smallest difference
    place10_diff = np.amin(np.floor(np.log10(np.abs(diffs))), axis=1).astype(int)
    # Round values to this power of 10 too
    pv_round = round_to_place(pv, place10_diff.reshape(-1, 1))
    # Power of 10 of the actual value
    place10_val = np.floor(np.log10(np.abs(pv[:, 1]))).astype(int)
    return pv_round, np.stack([place10_val, place10_diff], axis=1)

def color_best_outside_ci(row, clr="D0D0D0"):
    """ Check if the best value in the row is between lower and upper CIs;
    If not, return a string to add to each cell in that row of the LaTeX
    table, to make the cell a certain hexadecimal color clr.
    """
    if row.iat[1] < row.iat[0] or row.iat[1] > row.iat[2]:
        prefix = r"\cellcolor[HTML]{" + str(clr) + "}"
    else:
        prefix = ""
    return prefix


def format_float(v, rd):
    """ v is the value, rd is the number of decimal places """
    if (isinstance(rd, np.ndarray) or isinstance(rd, pd.Series)
        or isinstance(rd, pd.DataFrame) or isinstance(rd, list)):
        rd = rd[0]
    rdint = int(rd)
    return "$" + f"{v:.{rdint}f}" + "$"

def format_scinotation(v, rd):
    """ v is the value (CI or best), rd is the smallest power of 10 to
    keep and the power of the actual best value """
    place10_v, place10_rd = rd[0], rd[1]
    decimals = max(place10_v - place10_rd, 0)
    v_mantiss = v / 10**place10_v
    s = "$" + f"{v_mantiss:.{decimals}f}"
    s += r" \times 10^{" + str(place10_v) + "}$"
    return s

# midrules are put between each value of the block level
def write_tex_table(df, df_round, filename, filter_row, format_value, block_lvl):
    """ df_round can contain more than 1 column giving rounding info """
    if not isinstance(df_round, pd.DataFrame):
        df_round = df_round.to_frame()
    f = open(filename, "w")
    n_numeric = df.shape[1]
    n_alpha = len(df.index.names)
    lines = []
    lines.append("\\begin{tabular}{" + "l"*n_alpha + "c"*n_numeric + "}")
    lines.append("\\toprule")
    # Bolded header
    header = df.reset_index().columns
    header_strings = []
    for h in header:
        if h.startswith("$") and h.endswith("$"):
            header_strings.append(r"$\mathbf{" + h[1:-1] + "}$")
        else:
            header_strings.append("\\textbf{" + h + "}")
    header = "\t & ".join(header_strings) + r"  \\"
    lines.append(header)
    lines.append("\\midrule")

    # One block at a time. Add fake block level if there was none.
    ib = 0
    if block_lvl is not None:
        block_entries = df.index.get_level_values(block_lvl).unique()
        block_lvl_name = block_lvl
    else:  # Add a fake block level
        df = pd.concat({0:df}, names=["block_lvl"])
        df_round = pd.concat({0:df_round}, names=["block_lvl"])
        block_entries = [0]
        block_lvl_name = "block_lvl"

    for b in block_entries:
        # Make a multiline for the block label
        df_b = df.xs(b, level=block_lvl_name, axis=0)
        if block_lvl is not None:
            line = "\\multirow{" + str(df_b.shape[0]) + "}{*}{" + str(b) + "} \t & "
        else:
            line = ""
        # One row at a time, blank space after the first
        df_round_b = df_round.xs(b, level=block_lvl_name, axis=0)
        for il in range(df_b.shape[0]):
            if il > 0 and block_lvl is not None:
                line += "\t\t\t & "
            # Write the other index things
            for lvl in df_b.index.names:
                line += str(df_b.index.get_level_values(lvl)[il]) + " \t & "
            # Write the numerical values
            # Apply a user-provided filter for the values in this row
            # to determine any prefix, e.g. cell color, to apply to each cell
            cell_prefix = filter_row(df_b.iloc[il])
            for iv in range(df_b.shape[1]):
                v = df_b.iat[il, iv]
                rd = df_round_b.iloc[il, :].values
                line += cell_prefix + format_value(v, rd)
                if iv < df_b.shape[1]-1:
                    line += " \t & "

            # Finish the line
            line += r" \\"
            lines.append(line)
            # Set up the next one
            line = ""
        # Finish the block
        ib += 1
        if ib < len(block_entries):
            lines.append("\\midrule")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines = list(map(lambda x: x+ "\n", lines))
    lines[-1].strip("\n")
    f.writelines(lines)
    f.close()
    print("Wrote table to {}".format(filename))
    return 0


def main_mcmc_params():
    header_values = [percentile_symbol(2.5, place=1), "Best", percentile_symbol(97.5, place=1)]
    models_files_dict = {
        "6Y TCR/TCR, initial AKPR": "../results/mcmc/mcmc_analysis_shp1.json",
        "6Y TCR/TCR, revised AKPR": "../results/mcmc/mcmc_analysis_akpr_i.json",
        "6F TCR/TCR, revised AKPR": "../results/mcmc/mcmc_analysis_tcr_tcr_6f.json",
        "TCR/CAR": "../results/mcmc/mcmc_analysis_tcr_car_both_conc.json"
    }
    param_blocks = {}
    round_info = {}
    for mdl in models_files_dict:
        best_ci_arr, param_names = get_best_cis_names(models_files_dict[mdl])
        best_ci_arr, round_digits = round_best_cis(best_ci_arr)
        df = pd.DataFrame(
            best_ci_arr,
            index=pd.Index(param_names, name="Parameter"),
            columns=header_values
        )
        df_round = pd.Series(
            round_digits,
            index=pd.Index(param_names, name="Parameter")
        )
        param_blocks[mdl] = df
        round_info[mdl] = df_round
    param_blocks = pd.concat(param_blocks, names=["Model"], axis=0)
    round_info = pd.concat(round_info, names=["Model"])
    tex_table_filename = "../results/for_plots/mcmc_parameter_ci_table_auto.tex"

    write_tex_table(
        df=param_blocks,
        df_round=round_info,
        filename=tex_table_filename,
        filter_row=color_best_outside_ci,
        format_value=format_float,
        block_lvl="Model"
    )
    return 0


def main_surface_molec_numbers():
    surf_file = "../data/surface_counts/surface_molecule_summary_stats.h5"
    df_surface = pd.read_hdf(surf_file, key="surface_numbers_stats")

    latex_columns = ["Geometric mean CI 0.025", "Geometric mean", "Geometric mean CI 0.975"]
    ci_labels = [ci_symbol(2.5, 1), "Geo. mean", ci_symbol(97.5, 1)]

    df_surface = (df_surface.loc[:, latex_columns]
                    .rename(lambda x: x.replace("_", " "), level="Cell")
                    .rename({"B6":"B6 Splenocyte"}, level="Cell")
                    .rename(dict(zip(latex_columns, ci_labels)), axis=1)
                    .rename_axis(index={"Marker":"Molecule"})
                 )

    # Round to place of 95 % limit
    best_ci_arr, round_digits = round_best_cis_scinot(df_surface.values)
    df_surface_latex = pd.DataFrame(
        best_ci_arr,
        index=df_surface.index,
        columns=df_surface.columns
    )
    df_round_info = pd.DataFrame(
        round_digits,
        index=df_surface.index,
        columns=pd.Index(["place10_v", "place10_diff"], name="Info")
    )

    tex_table_filename = "../results/for_plots/surface_molecule_summary_table_auto.tex"
    write_tex_table(
        df=df_surface_latex,
        df_round=df_round_info,
        filename=tex_table_filename,
        filter_row=color_best_outside_ci,
        format_value=format_scinotation,
        block_lvl=None
    )
    return 0


if __name__ == "__main__":
    main_mcmc_params()
    main_surface_molec_numbers()
