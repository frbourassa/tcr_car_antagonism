"""
Module with simple data manipulation and preprocessing functions.

@author: frbourassa
May 2023
"""
import numpy as np
import pandas as pd

# Global parameters, useful constants
ln10 = np.log(10.0)
eps_for_log = 1e-8

# Group by all levels but lvls
def groupby_others(df, lvls, axis=0):
    idx = df.index if axis == 0 else df.columns
    nms = list(idx.names)
    for lv in lvls:
        try:
            nms.remove(lv)
        except ValueError:
            raise KeyError("{} not in index names {}".format(lv, idx.names))
    df2 = df.groupby(nms, axis=axis)
    return df2

# Geometric mean calculation
def geo_mean(df, axis=1):
    return np.exp(np.log(df).mean(axis=axis))


def geo_mean_apply(ser):
    return np.exp(np.mean(np.log(ser)))


def geo_mean_levels(df, lvls, axis=0):
    df_groups = groupby_others(np.log(df), lvls, axis=axis)
    return np.exp(df_groups.mean())


# Function for timing format
def time_dd_hh_mm_ss(t):
    days = int(t // (3600*24))
    t2 = t - days*24*3600
    hours = int(t2 // 3600)
    t2 -= hours*3600
    minutes = int(t2 // 60)
    t2 -= minutes*60
    seconds = int(t2 // 1)
    t2 -= seconds*1
    milli = int(t2*1000)

    s = "{} d {:0>2d}h{:0>2d}m{:0>2d}.{:0>3d}s".format(
                            days, hours, minutes, seconds, milli)
    return s



# Functions for concentration conversions
def hill(x, a, h, n):
    normx = (x / h)**n
    return a * normx / (1.0 + normx)

def michaelis_menten(x, a, h):
    normx = (x / h)
    return a * normx / (1.0 + normx)

def loglog_michaelis_menten(logx, a, h):
    logh = np.log(h)
    return np.log(a * np.exp(logx - logh) / (1.0 + np.exp(logx - logh)))

def inverse_michaelis_menten(y, a, h):
    normy = y / a
    return h*(normy / (1.0 - normy))


def read_conc_uM(text):
    """ Convert a concentration value in text into a float (M units). """
    dico_units = {
        'f':1e-15,   # femto
        "p": 1e-12,  # pico
        'n':1e-9,    # nano
        'u':1e-6,    # micro
        'm':1e-3,    # milli
        'c':1e-2,    # centi
        'd':0.1,     # deci
        '':1.
    }
    # Separate numbers from letters
    value, units = [], []
    for a in text:
        if a.isnumeric():
            value.append(a)
        elif a.isalpha():
            units.append(a)

    # Put letters together and numbers together
    value = ''.join(value)
    units = ''.join(units)
    conc = float(value)

    # Read the order of magnitude
    units = units.replace("M", '')

    # If we encounter a problem here, put a value that is impossibly large
    if len(units) == 1:
        conc *= dico_units.get(units, 1e6)
    else:
        conc = 1e6

    return conc*1e6


# Function to revert ligand numbers to pulse concentration strings.
def write_conc_uM(c):
    """ Convert a concentration value in uM units into a string label. """
    dico_units_reverse = {
        -18: "a",   # atto
        -15: 'f',   # femto
        -12: 'p',   # pico
        -9: 'n',    # nano
        -6: 'u',    # micro
        -3: 'm',    # milli
        0: ''
    }
    # Convert to M
    c = c * 1e-6 * 1.001  # Account for small conversion errors.

    # Find closest lower power of 10
    power_10 = int(np.log10(c) // 1)
    # Find closest lower power in dico_units_reverse's keys
    power_units = -np.inf
    for k in dico_units_reverse.keys():
        if k <= power_10 and k > power_units:
            power_units = k
    # Remainder: factor of 1, 10 or 100 above the unit
    remain = power_10 - power_units  # May be 0, 1, or 2
    prefactor = int(10**remain)

    return str(prefactor) + dico_units_reverse[power_units] + "M"


# Other string conversion utility, for k, m, f labels in our case
def string_to_tuple(s):
    """ Convert a string back into the original tuple of ints """
    return tuple(int(x) for x in s.strip("()").split(",") if x != "")
