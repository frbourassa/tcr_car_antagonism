"""
Module with functions to convert experimental measurements into
into model parameters, for instance antigen EC50s into ligand binding times
or pulse concentrations into absolute ligand numbers.

@author: frbourassa
May 2023
"""

### EC50 TO TAU CONVERSION
# Version based on a set threshold
def convert_ec50_tau_thresh(ec50, thresh, npow):
    """ Compute tau associated to an EC50, given a KPR threshold """
    return (thresh / ec50)**(1.0/npow)


# Version based on a relative EC50 and a reference tau
def convert_ec50_tau_relative(ec50rel, reftau, npow):
    """ Compute tau associated to an EC50, given a reference tau
    and the EC50 relative to that tau (EC50 / EC50ref). """
    return reftau / ec50rel**(1.0 / npow)


### TAU TO EC50 CONVERSION
# Version based on a set threshold
def convert_tau_ec50_thresh(tau, thresh_ec50, npow):
    """ Compute EC50 associated to a tau, given a KPR threshold """
    return thresh_ec50 / tau**npow

# Version based on a relative EC50 and a reference tau
def convert_tau_ec50_relative(tau, reftau, npow):
    """ Compute relative EC50 associated to a tau, given a reference tau. """
    return (reftau / tau) ** npow
