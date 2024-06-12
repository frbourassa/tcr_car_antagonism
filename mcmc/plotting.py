""" Analysis tools producing generic plots to assess the convergence
and quality of parameter estimation algorithm results, like MCMC.

@author: frbourassa
May 2022
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.text as mtext
from matplotlib.legend_handler import HandlerBase
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
import itertools


# For adding subtitles in legends.
# Artist class: handle, containing a string
class LegendSubtitle(object):
    def __init__(self, message, **text_properties):
        self.text = message
        self.text_props = text_properties
        text_len = max(map(len, self.text.split("\n")))
        self.labelwidth = " "*int(text_len*1.33) + "\n"*self.text.count("\n")
    def get_label(self, *args, **kwargs):
        return self.labelwidth  # no label, the artist itself is the text

# Handler class, give it text properties
class LegendSubtitleHandler(HandlerBase):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        title = mtext.Text(x0, y0, orig_handle.text, size=fontsize,
                    ha="left", va="center", **orig_handle.text_props)
        handlebox.add_artist(title)
        # Make the legend box wider if needed
        return title


# Utility for log ticks in scientific notation
def change_log_ticks(axis, base=2, which="y"):
    if which == "y" or which == "both":
        ogyticks = axis.get_yticks()
        newyticks = list(np.unique([int(x) for x in ogyticks]))
        newyticklabels = [str(base)+'$^{'+str(x)+'}$' for x in newyticks]
        axis.set_yticks(newyticks)
        axis.set_yticklabels(newyticklabels)
    if which == "x" or which == "both":
        ogxticks = axis.get_xticks()
        newxticks = list(np.unique([int(x) for x in ogxticks]))
        newxticklabels = [str(base)+'$^{'+str(x)+'}$' for x in newxticks]
        axis.set_xticks(newxticks)
        axis.set_xticklabels(newxticklabels)
    return axis


### Automatic style generation functions
# Replace old core seaborn functions, which were moved
# around in the package since then
def unique_markers(n):
    """Build an arbitrarily long list of unique marker styles for points.
    Parameters
    ----------
    n : int
        Number of unique marker specs to generate.
    Returns
    -------
    markers : list of string or tuples
        Values for defining :class:`matplotlib.markers.MarkerStyle` objects.
        All markers will be filled.
    """
    # Start with marker specs that are well distinguishable
    markers = [
        "o",
        "s",
        "X",
        (4, 0, 0),
        "^",
        "v",
        (4, 1, 45),
    ]

    # Now generate more from regular polygons of increasing order
    s = 5
    while len(markers) < n:
        a = 360 / (s + 1) / 2
        markers.extend([
            (s + 1, 1, a),
            (s + 1, 0, a),
            (s, 1, 0),
            (s, 0, 0),
        ])
        s += 1

    # Convert to MarkerStyle object, using only exactly what we need
    # markers = [mpl.markers.MarkerStyle(m) for m in markers[:n]]

    return markers[:n]


def unique_dashes(n):
    """Build an arbitrarily long list of unique dash styles for lines.
    Parameters
    ----------
    n : int
        Number of unique dash specs to generate.
    Returns
    -------
    dashes : list of strings or tuples
        Valid arguments for the ``dashes`` parameter on
        :class:`matplotlib.lines.Line2D`. The first spec is a solid
        line (``""``), the remainder are sequences of long and short
        dashes.
    """
    # Start with dash specs that are well distinguishable
    dashes = [
        "",
        (4, 1.5),
        (1, 1),
        (3, 1.25, 1.5, 1.25),
        (5, 1, 1, 1),
    ]

    # Now programatically build as many as we need
    p = 3
    while len(dashes) < n:

        # Take combinations of long and short dashes
        a = itertools.combinations_with_replacement([3, 1.25], p)
        b = itertools.combinations_with_replacement([4, 1], p)

        # Interleave the combinations, reversing one of the streams
        segment_list = itertools.chain(*zip(
            list(a)[1:-1][::-1],
            list(b)[1:-1]
        ))

        # Now insert the gaps
        for segments in segment_list:
            gap = min(segments)
            spec = tuple(itertools.chain(*((seg, gap) for seg in segments)))
            dashes.append(spec)

        p += 1

    return dashes[:n]


### General legend generation functions ###

def handles_properties_legend(hues, styles, sizes):
    """ Creates handles (with labels) for a categorical legend with a section
    for hues, one for styles and markers, one for sizes.
    Pass None to skip that legend section.
    Args:
        hues (tuple): hue_title, hue_entries
            hue_entries: dictionary of label: hue
        styles (tuple): style_title, style_entries
            style_entries: dictionary of label: (linestyle, marker)
        sizes (tuple): size_title, size_entries
            size_entries: dictionary of label: (linesize, markersize)
    Returns:
        legend_handles
        handler_map

    """
    legend_handles = []
    legend_handler_map = {LegendSubtitle: LegendSubtitleHandler()}

    # Hues section
    if hues is not None:
        hue_title, hue_dict = hues
        legend_handles.append(LegendSubtitle(hue_title))
        for lbl in hue_dict:
            legend_handles.append(Line2D([0], [0], color=hue_dict[lbl],
                label=lbl, linestyle="-"))

    # Line styles section
    if styles is not None:
        style_title, style_dict = styles
        legend_handles.append(LegendSubtitle(style_title))
        for lbl in style_dict:
            legend_handles.append(Line2D([0], [0], color="k", label=lbl,
                linestyle=style_dict[lbl][0], marker=style_dict[lbl][1],
                mfc="k", mec="k"))
    if sizes is not None:
        size_title, size_dict = sizes
        legend_handles.append(LegendSubtitle(size_title))
        for lbl in size_dict:
            legend_handles.append(Line2D([0], [0], color="k", label=lbl,
                linestyle="-", marker="o", mfc="k", mec="k",
                linesize=size_dict[lbl][0], markersize=size_dict[lbl][1]))
    return legend_handles, legend_handler_map


def prepare_hues(df, lvl, sortkws={}, colmap=None):
    """ Prepare color palette for the entries in the level lvl of
    df. Sort them according to sortkey. Use colors from colmap.
    """
    try:
        hue_entries = list(df.index.get_level_values(lvl).unique())
    except ValueError:
        palette = {}
        hue_entries = []
    else:
        hue_entries = sorted(hue_entries, **sortkws)
        palette = sns.color_palette(colmap, n_colors=len(hue_entries))
        palette = {hue_entries[i]:palette[i] for i in range(len(hue_entries))}
    return hue_entries, palette


def prepare_styles(df, lvl, sortkws={}):
    """ Prepare dictionary of line styles for the entries in the level lvl
    of df. Sort them according to sortkey.
    """
    try:
        style_entries = list(df.index.get_level_values(lvl).unique())
    except KeyError:
        stdict = {}
        style_entries = []
    else:
        style_entries = sorted(style_entries, **sortkws)
        stdict = unique_dashes(len(style_entries))
        stdict = {style_entries[i]:(0, stdict[i]) for i in range(len(style_entries))}
    return style_entries, stdict


def prepare_markers(df, lvl, sortkws={}):
    """ Prepare dictionary of line styles for the entries in the level lvl
    of df. Sort them according to sortkey.
    """
    try:
        marker_entries = list(df.index.get_level_values(lvl).unique())
    except KeyError:
        markdict = {}
        marker_entries = []
    else:
        marker_entries = sorted(marker_entries, **sortkws)
        markdict = unique_markers(len(marker_entries))
        markdict = {marker_entries[i]:markdict[i] for i in range(len(marker_entries))}
    return marker_entries, markdict


def prepare_subplots(df, row_lvl, col_lvl, sortkws_row={},
                sortkws_col={}, **kwargs):
    """ Extra kwargs are passed to plt.subplots """
    try:
        row_vals = list(df.index.get_level_values(row_lvl).unique())
    except KeyError:
        nrows = 1
        row_vals = []
    else:
        row_vals = sorted(row_vals, **sortkws_row)
        nrows = len(row_vals)

    try:
        col_vals = list(df.index.get_level_values(col_lvl).unique())
    except KeyError:
        ncols = 1
        col_vals = []
    else:
        col_vals = sorted(col_vals, **sortkws_col)
        ncols = len(col_vals)
    fig, axes = plt.subplots(nrows, ncols, **kwargs)
    # Make sure axes is a 2d array
    axes = np.atleast_2d(axes)
    return row_vals, col_vals, fig, axes


### Specific legend creation function for data vs model fits
def data_model_handles_legend(palette, markers, styles, palette_name,
            model_style="-", model_lw=2.0, data_size=8, data_marker="o"):
    """ Prepare a list of legend handles with legend Subtitles,
    return a legend handler map too

    Args:
        palette (dict): {condition: color} dictionary
        markers (dict): {condition: marker} dictionary of data points
        styles (str): line style of model curves
        palette_name (str): name of the property delineated
            by palette/markers/styles

    Note: the palette, markers, and styles dictionaries are supposed to
        have the same keys.

    Returns:
        legend_handes, legend_handler_map
    """
    legend_handles = []
    legend_handler_map = {LegendSubtitle: LegendSubtitleHandler()}

    # First, clarify data vs model
    legend_handles.append(Line2D([0], [0], ls="none", marker=data_marker,
                mfc="grey", mec="grey", label="Data", ms=data_size))
    legend_handles.append(Line2D([0], [0], ls=model_style,
                color="grey", marker=None, label="Model", lw=model_lw))

    # Colors, linestyles and markers together
    # TODO: ideally, the marker and line would be separated, e.g. with marker
    # above the line, rather than stacked.
    legend_handles.append(LegendSubtitle(palette_name))
    for k in palette.keys():
        color = palette.get(k)
        legend_handles.append(Line2D([0], [0], ls=styles.get(k),
            marker=markers.get(k), color=color, mfc=color,
            mec=color, ms=data_size, lw=model_lw, label=k))

    return legend_handles, legend_handler_map

# Hexbin plot for use with pairgrid taken from StackOverflow:
# https://stackoverflow.com/questions/40495093/hexbin-plot-in-pairgrid-with-seaborn
def hexbin(x, y, color, max_ser=None, min_ser=None, **kwargs):
    cmap = sns.light_palette(color, as_cmap=True)
    ax = plt.gca()
    xmin, xmax = min_ser[x.name], max_ser[x.name]
    ymin, ymax = min_ser[y.name], max_ser[y.name]
    artist = ax.hexbin(x, y, gridsize=15, cmap=cmap,
            extent=[xmin, xmax, ymin, ymax], **kwargs)
    return artist


## Pairplot with hexbins off-diagonal
def hexpair(data, grid_kws={}, hexbin_kws={}, diag_kws={}):
    """ Wrapper around FacetGrid, mapping sns.jointplot to lower
    half and histogram to diagonal. grid_kws contain arguments about
    the variables to be plotted, hues, and so on.

    Args:
        data (pd.DataFrame)
        grid_kws (dict): keyword arguments passed to seaborn.PairGrid.
            Accepted values and defaults: hue=None, hue_order=None,
            palette=None, hue_kws=None, vars=None, x_vars=None, y_vars=None,
            corner=False, diag_sharey=True, height=2.5, aspect=1,
            layout_pad=0.5, despine=True, dropna=False, size=None

        hexbin_kws (dict): keyword arguments passed to sns.jointplot,
            apart from kind="hex". Accepted values and defaults:


        diag_kws (dict): keyword arguments passed to sns.histogram.
            Accepted values and defaults: hue=None, weights=None, stat='count',
            bins='auto', binwidth=None, binrange=None, discrete=None,
            cumulative=False, common_bins=True, common_norm=True,
            multiple='layer', element='bars', fill=True, shrink=1, kde=False,
            kde_kws=None, line_kws=None, thresh=0, pthresh=None, pmax=None,
            cbar=False, cbar_ax=None, cbar_kws=None, palette=None,
            hue_order=None, hue_norm=None, color=None, log_scale=None,
            legend=True, ax=None, **kwargs

    Returns:
        g (sns.PairGrid)
    """
    g = sns.PairGrid(data, **grid_kws)
    g = g.map_diag(sns.histplot, **diag_kws)
    g = g.map_lower(hexbin, min_ser=data.min(), max_ser=data.max(), **hexbin_kws)
    for i in range(g.axes.shape[0]):
        g.axes[i, i].set_ylabel("Density")
        for j in range(i+1, g.axes.shape[1]):
            g.axes[i, j].set_axis_off()
    return g


def average_time_series(t, y):
    # Use the trapeze rule with potentially unequal time intervals
    dt = np.diff(t)
    return np.sum((y[1:] + y[:-1])*dt*0.5) / (t.max() - t.min())


if __name__ == "__main__":
    # Test hexpair plot with dummy data
    import pandas as pd
    vars = ["w", "x", "y", "z"]
    numbers, letters = np.arange(100), list("abcd")
    idx = pd.MultiIndex.from_product([numbers, letters],
            names=["Number", "Letter"])
    cols = pd.Index(vars, name="Variable")
    rgen = np.random.default_rng(seed=0xf6478c6135e8ff7cfca9cacb35ff916b)
    arr = rgen.standard_normal(size=(len(idx), len(cols)))
    dat = pd.DataFrame(arr, index=idx, columns=cols)

    gg = hexpair(
        data=dat.reset_index(),
        grid_kws={
            "hue": "Letter",
            "hue_order": letters[::-1],
            "palette": sns.color_palette("magma", n_colors=len(letters)),
            "vars": vars[1:],
            "layout_pad": 0.4
        },
        hexbin_kws={
            "alpha": 0.5
        },
        diag_kws={
            "legend": False
        })
    plt.show()
    plt.close()
