import pandas as pd
import argparse

def multi_replace(s, dico):
    """ dict contains pairs of 'old':'new' substrings """
    for o in dico:
        s = s.replace(o, dico[o])
    return s

def reformat_names(idx):
    substitutions = {
        " ": "_",
        "#": "nb"
    }
    if isinstance(idx, pd.MultiIndex):
        new_names = []
        for n in idx.names:
            new_names.append(multi_replace(n, substitutions))
        idx = idx.set_names(new_names)

    # In any case, check axis' name itself
    new_name = multi_replace(str(idx.name), substitutions)
    idx.name = new_name
    return idx

def pickle_to_hdf(fname, key="df"):
    if not fname.endswith(".pkl"):
        raise TypeError("Not a .pkl file; use the right extension.")
    df = pd.read_pickle(fname)
    # Convert index and columns names to acceptable strings
    # Currently, the onlt use case covered is replacing spaces by _
    df.index = reformat_names(df.index)
    df.columns = reformat_names(df.columns)
    df.to_hdf(fname.replace(".pkl", ".hdf"), key=key)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=(
        'Transform a pickle file to hdf with proper index level names.'
         + 'Run using a recent version of pickle if necessary'))
    parser.add_argument('files', action="store", metavar='files', type=str, nargs='+',
                    help='a file name for the list of files to treat')
    args = parser.parse_args()

    for f in args.files:
        try:
            pickle_to_hdf(f)
        except Exception as e:
            print("Could not process file {} due to error:".format(f))
            print(e)
