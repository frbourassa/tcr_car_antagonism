"""
@author: frbourassa
2023
"""

import os, json

# Data wrangling functions
def strip_names(df, char, axis=0):
    stripper = lambda x: x.strip(char)
    return df.rename(mapper=stripper, axis=axis)


def pad_names(df, char, axis=0):
    padder = lambda x: char + x + char
    return df.rename(mapper=padder, axis=axis)


# Data import and export functions
def dict_to_hdf5(gp, di):
    """ Store dictionary di values as datasets in hdf5 group gp"""
    for k in di.keys():
        gp[k] = di[k]
    return gp


def hdf5_to_dict(gp):
    """ Retrieve datasets of hdf5 group gp into a dictionary"""
    di = {}
    for k in gp:
        di[k] = gp.get(k)[()]
    return di


def save_defaults_json(di, fpath, overwrite=False):
    """ Save a dictionary to a JSON file, but if the file already exists
    update it with values for keys not already in it, without replacing
    existing values.
    """
    # File does not exist or can be overwritten
    if not os.path.isfile(fpath) or overwrite:
        with open(fpath, "w") as file:
            json.dump(di, file, indent=2)
        di_file = di
    else:
        with open(fpath, "r") as file:
            di_file = json.load(file)
        with open(fpath, "w") as file:
            for k in di.keys():
                di_file.setdefault(k, di[k])
            json.dump(di_file, file, indent=2)

        # If performance was important, we should use builtins only.
        # But side effect: the input dict is modified in-place, so avoid.
        #di.update(di_file)
        #json.dump(di, file, indent=2)

    return di_file


# Printing functions
def nice_dict_print(di):
    for k in di.keys():
        print(str(k) + ":", di[k])
    return di


# Reconstruct objects with .slope, .intercept, etc. attributes from saved dict.
class LinRegRes():
    def __init__(self, di):
        for k in di:
            setattr(self, k, di[k])
