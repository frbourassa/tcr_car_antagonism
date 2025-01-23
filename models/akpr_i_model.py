# -*- coding:utf-8 -*-
""" Module containing functions to solve for the steady-state of the
improved AKPR model."""

import numpy as np
from scipy.optimize import root_scalar


## Solving for C_T and D_T numerically (without the non-saturation approximation)
# Without antagonism, solution of f(L) = 0
def solution_CT(Lp, taup, kappap, Rp):
    """ Solves the quadratic equation for C_T without antagonism.
    Can also be used vectorially with L1p, tau1p, kappap, Rp being arrays
    of parameter values for the different receptor types.
    """
    bterm = Lp + Rp + 1/(kappap*taup)
    return 0.5 * bterm - 0.5*np.sqrt(bterm**2 - 4*Lp*Rp)


# This solves for C_T and D_T, switching to a simple quadratic equation if L2 is 0.
def solve_CT_DT(lsp, tausp, kappap, Rp):
    """ Use the np.roots function to solve the cubic equation
    that results for C_T after substitution of D_T(C_T).
    The roots are found within machine precision,
    since there is an exact formula for cubic. """
    # If there is a single ligand, really:
    L1p, L2p = lsp
    tau1p, tau2p = tausp
    if L2p < 1e-12:
        c_t, d_t = solution_CT(L1p, tau1p, kappap, Rp), 0.
        return c_t, d_t

    # Solving the first equation for D_T(C_T) and inserting in the second, we get
    # a cubic equation for C_T with coefficients
    p = np.zeros(4)
    p[0] = tau1p / tau2p - 1.
    p[1] = Rp + L1p - L2p - tau1p/tau2p * (Rp + 2.*L1p) - 1. / (kappap * tau2p) + 1. / (kappap * tau1p)
    p[2] = tau1p / tau2p * L1p * (2.*Rp + L1p) + L1p / (kappap * tau2p) - Rp * L1p + L1p * L2p
    p[3] = -tau1p / tau2p * Rp * L1p**2
    c_ts = np.roots(p)

    # Keep only real roots that are in [0, min(L_1, R)]
    isreal = (np.abs(np.imag(c_ts)) < 1e-14)
    isrange = (np.real(c_ts) >= 0.)*(np.real(c_ts) <= min(Rp, L1p))
    c_ts = np.real(c_ts[np.logical_and(isreal, isrange).astype(bool)])

    # There should already be a single root left in the correct range
    if len(c_ts) > 1:
        print("Many C_T solutions in the correct range:", c_ts)

    # Lastly, enforce D_T in the correct range too, to make sure.
    d_ts = Rp - c_ts - c_ts / (kappap * tau1p * (L1p - c_ts))
    d_isrange = ((d_ts >= 0.)*(d_ts <= min(Rp, L2p))).astype(bool)

    if np.sum(d_isrange) != 1:
        print("There are either no roots or too many; " +
        "we found D_T = {} when L_1 = {}, L_2={}, R={}".format(d_ts, L1p, L2p, Rp))
        d_ts = np.clip(d_ts, a_min=0.0, a_max=min(Rp, L2p))
        d_isrange = True
    c_ts = float(c_ts[d_isrange][0])
    d_ts = float(d_ts[d_isrange][0])
    return c_ts, d_ts  # C_T and D_T


## Negative feedback functions: psi(S), and solving for S in implicit cases
def psi_of_i(i, ithresh, ki, phip, psi0):
    return phip / (1.0 + (i / ithresh)**ki) + psi0


# Solving implicit equation for I when m >= N - f
def equation_i_1ligand(i, ratesp, tau1p, ctot, itot, nmf):
    phip, kappap, Csp, I0p, kp, psi0 = ratesp
    geofact = phip*tau1p / (phip*tau1p + 1.0)
    psi_i = psi_of_i(i, I0p, kp, phip, psi0)

    # Compute C_m given the tentative I
    # If m == N - f, rate into it is phi, out of it is psi
    n_minus_f = nmf[0] - nmf[2]
    cm = geofact**n_minus_f * ctot / (psi_i*tau1p + 1.0)
    # If m is larger than that, but smaller than N, this is
    # some complex between N-f and the last
    if nmf[1] > n_minus_f:
        n_extra = min(nmf[1] - n_minus_f, nmf[2]-1)  # In case m == N, clip to f-1
        geofact2 = psi_i*tau1p / (psi_i*tau1p + 1.0)
        cm = cm * geofact2**n_extra
        if nmf[1] == nmf[0]:  # Last complex: special factor
            cm = cm * psi_i*tau1p

    # Compute the activation function F in I = F(C_m(I))
    activ_fct_i = itot * cm / (cm + Csp)
    return i - activ_fct_i


# Solving implicit equation for I when m >= N - f
def equation_i_2ligands(i, ratesp, tausp, c_d_tot, itot, nmf):
    phip, kappap, Csp, I0p, kp, psi0 = ratesp
    tau1p, tau2p = tausp
    geofact1 = phip*tau1p / (phip*tau1p + 1.0)
    geofact2 = phip*tau2p / (phip*tau2p + 1.0)
    psi_i = psi_of_i(i, I0p, kp, phip, psi0)
    ctot, dtot = c_d_tot

    # Compute C_m and D_m given the tentative s
    # If m == N - f, rate into it is phi, out of it is psi
    n_minus_f = nmf[0] - nmf[2]
    cm = geofact1**n_minus_f * ctot / (psi_i*tau1p + 1.0)
    dm = geofact2**n_minus_f * dtot / (psi_i*tau2p + 1.0)
    # If m is larger than that, but smaller than N, this is
    # some complex between N-f and the last
    if nmf[1] > n_minus_f:
        n_extra = min(nmf[1] - n_minus_f, nmf[2]-1)  # In case m == N, clip to f-1
        geofact12 = psi_i*tau1p / (psi_i*tau1p + 1.0)
        geofact22 = psi_i*tau2p / (psi_i*tau2p + 1.0)
        cm = cm * geofact12**n_extra
        dm = dm * geofact22**n_extra
        if nmf[1] == nmf[0]:  # Last complex: special factor
            cm = cm * psi_i*tau1p
            dm = dm * psi_i*tau2p
    cdm = cm + dm

    # Compute the activation function F in I = F(C_m(I))
    activ_fct_i = itot * cdm / (cdm + ratesp[2])
    return i - activ_fct_i


## Main model functions
# Special version of the AKPR with I model solution with a single ligand, faster to compute.
def steady_akpr_i_1ligand(ratesp, tau1p, L1p, ri_tots, nmf, large_l=False):
    """ Solving for the steady_state of the I model for a single ligand type.

    Args:
        ratesp (list): phi, kappa, beta/alpha, I0p, kp, phi0
        tau1p (float): binding time of the agonists
        L1p (float): number of agonist ligands
        ri_tots (list of 2 floats): [R_tot, I_tot]
            total number of receptors
            total number of inhibitory molecules (hint: keep to one)
        nmf (list of 3 ints): [N, m, f]
            Np (int): the number of phosphorylated complexes in the cascade
            mp (int): complex mediating negative feedback (I phosphorylation)
            fp (int): last f steps receive negative feedback.
        large_l (bool): if True, take L_F -> infinity, implying C_T=R
    Returns:
        complexes (np.ndarray): 1D array of complexes, ordered as
            [C_0, C_1, ..., C_N, S]
    """
    # Rates
    phip, kappap, Csp, I0p, kp, psi0 = ratesp
    Rp, ITp = ri_tots
    Np, mp, fp = nmf
    #if mp >= Np - fp:
    #    raise NotImplementedError("Can't use m >= N-f, it requires a numerical solution")

    # Solve for C_T and D_T
    if large_l:
        CT = Rp
    else:
        CT = solution_CT(L1p, tau1p, kappap, Rp)
    #CT = kappa*Rp*L1p*tau1p / (kappa*Rp*tau1p + 1.0)

    # Quantities that come back often
    geofact = phip*tau1p / (phip*tau1p + 1.0)

    # Compute all Cns and Dns below N-f: apply same factor recursively
    # Note: if N-f = 0, complexes[0] will be changed below, that's OK
    complexes = np.zeros(Np + 2)
    complexes[0] = CT / (phip*tau1p + 1.0)
    for n in range(1, Np-fp):
        complexes[n] = complexes[n-1]*geofact

    # Now, compute I and psi(I)
    # If mp < Np - fp, we have already computed C_m, this is feedforward.
    if mp < Np - fp:
        complexes[-1] = ITp * complexes[mp] / (complexes[mp] + Csp)
    # Else, we need to determine I numerically from an implicit equation
    # I = F(C_m(I)).
    else:
        res = root_scalar(equation_i_1ligand, args=(ratesp, tau1p, CT, ITp, nmf),
                                   bracket=[0.0, ITp], x0=ITp/2.0, rtol=1e-6)
        if res.converged:
            complexes[-1] = res.root
        else:
            raise RuntimeError("I did not converge; flag {}".format(res.flag))

    # From S, compute the phosphorylation of the last f+1 complexes
    psi_i = psi_of_i(complexes[-1], I0p, kp, phip, psi0)

    # Finally, compute C_{N-f} to C_N
    if Np - fp > 0:
        complexes[Np-fp] = complexes[Np-fp-1] * phip*tau1p / (psi_i*tau1p + 1.0)
    elif Np - fp == 0:  # change C_0 to correct expression
        complexes[Np-fp] = CT / (psi_i*tau1p + 1.0)
    geofact = psi_i*tau1p / (psi_i*tau1p + 1.0)
    for n in range(Np-fp+1, Np):
        complexes[n] = complexes[n-1] * geofact
    complexes[Np] = complexes[Np-1] * psi_i*tau1p

    return complexes


def steady_akpr_i_2ligands(ratesp, tausp, lsp, ri_tots, nmf):
    """ Solving for the steady_state of the I model with a mixture of
    two ligands (or only one by inputting L2p=0).

    Args:
        ratesp (list): phi, kappa, beta/alpha, I0p, kp, psi0
        tausp (list of 2 floats): binding times of the ligands
        lsp (list of 2 floats): [L1, L2], numbers of ligands
        ri_tots (list of 2 floats): [R_tot, I_tot]
            total number of receptors
            total number of phosphatase molecules (hint: keep to one)
        nmf (list of 3 ints): [N, m, f]
            Np (int): the number of phosphorylated complexes in the cascade
            mp (int): complex mediating negative feedback (I phosphorylation)
            fp (int): last f steps receive negative feedback.
    Returns:
        complexes (np.ndarray): 1D array of complexes, ordered as
            [C_0, C_1, ..., C_N, D_0, D_1, ..., D_N, S]
    """
    # Rates
    phip, kappap, Csp, I0p, kp, psi0 = ratesp
    Rp, ITp = ri_tots
    Np, mp, fp = nmf
    tau1p, tau2p = tausp
    #if mp >= Np - fp:
    #    raise NotImplementedError("Can't use m >= N-f, it requires a numerical solution")

    # Solve for C_T and D_T
    CT, DT = solve_CT_DT(lsp, tausp, kappap, Rp)
    # CT = kappa*Rp*lsp[0]*tau1p / (kappa*Rp*tau1p + 1.0)
    # DT = kappa*Rp*lsp[1]*tau2p / (kappa*Rp*tau2p + 1.0)

    # Quantities that come back often
    geofact1 = phip*tau1p / (phip*tau1p + 1.0)
    geofact2 = phip*tau2p / (phip*tau2p + 1.0)

    # Compute all Cns and Dns below N-f: apply same factor recursively
    # Note: if N-f = 0, complexes[0] and Np+1 will be changed below, that's OK
    complexes = np.zeros(2*Np + 3)  # 2*(N+1) + I
    complexes[0] = CT / (phip*tau1p + 1.0)
    complexes[Np+1] = DT / (phip*tau2p + 1.0)
    for n in range(1, Np-fp):
        complexes[n] = complexes[n-1]*geofact1
        complexes[Np+1+n] = complexes[Np+n]*geofact2

    # Now, compute I and psi(I)
        # Now, compute I and psi(I)
    # If mp < Np - fp, we have already computed C_m, this is feedforward.
    if mp < Np - fp:
        cdm = complexes[mp] + complexes[mp+Np+1]
        complexes[-1] = ITp * cdm / (cdm + Csp)
    # Else, we need to determine I numerically from an implicit equation
    # I = F(C_m(I)).
    else:
        res = root_scalar(equation_i_2ligands, args=(ratesp, tausp, [CT, DT], ITp, nmf),
                                   bracket=[0.0, ITp], x0=ITp/2.0, rtol=1e-6)
        if res.converged:
            complexes[-1] = res.root
        else:
            raise RuntimeError("I solution did not converge; flag {}".format(res.flag))

    psi_i = psi_of_i(complexes[-1], I0p, kp, phip, psi0)

    # Finally, compute C_{N-f}, D_{N-f} to C_N and D_N
    if Np - fp > 0:
        complexes[Np-fp] = complexes[Np-fp-1] * phip*tau1p / (psi_i*tau1p + 1.0)
        complexes[2*Np-fp+1] = complexes[2*Np-fp] * phip*tau2p / (psi_i*tau2p + 1.0)
    else:  # Correct C_0, D_0
        complexes[Np-fp] = CT / (psi_i*tau1p + 1.0)
        complexes[2*Np-fp+1] = DT / (psi_i*tau2p + 1.0)
    geofact1 = psi_i*tau1p/ (psi_i*tau1p + 1.0)
    geofact2 = psi_i*tau2p/ (psi_i*tau2p + 1.0)
    for n in range(Np-fp+1, Np):
        complexes[n] = complexes[n-1]*geofact1
        complexes[n+Np+1] = complexes[n+Np]*geofact2

    complexes[Np] = complexes[Np-1] * psi_i*tau1p
    complexes[2*Np+1] = complexes[2*Np] * psi_i*tau2p

    return complexes


## Output thresholding for comparison of types
## Calculating the output of each receptor type
# Computing the threshold of activation theta_n for a receptor type
# The threshold should be roughly speaking C_N = 1. We tune model parameters
# such that
def get_threshold(nparams, ratep, ri_tots, tau_c=3.0):
    """
    Args:
        nparam (list of 3 ints): N, m, f
        ratep (list): phi, kappa, cmthresh, sthresh, k, gamma, psi0  for one receptor type.
        ri_tots (list of 2 floats): [R_tot, I_tot]
        tau_c (float): tau value for the threshold
    L is set equal to 10x the number of receptors: response at saturation.
    """
    # Thresholds for C_N
    thresh = steady_akpr_i_1ligand(ratep, tau_c, ri_tots[0]*10, ri_tots,
                            nparams, large_l=True)[nparams[0]]
    return thresh


def activation_function(c, thresh, pwr=1):
    #return (c / thresh)**pwr
    return c**pwr / (c**pwr + thresh**pwr)


# total output of the receptor type, given threshold on the C_N
# and the output of steady_akpr_i_receptor_types
def compute_zoutput_from_complexes(complexes, threshold, Np):
    r""" Compute \sum_i Z_i, where Z_i = C^i_N/(C^i_N + \theta^i_N) is the output
    of receptor type i.
    """
    cn_plus_dn = complexes[Np] + complexes[2*Np+1]
    return activation_function(cn_plus_dn, threshold)


### IL-2 MODEL ###
# Simplification of the Voisinne 2015 model, no modifications to
# APC dynamics. T cell number assumed constant. The real dynamics
# are approximated by the slow decay of antigens on APCs, this gives
# effective IL-2 dynamics that are correct, so why bother at this stage?
def jak(r, i2, ec50lo, ec50hi, r0, base):
    mm_r = r / (r + r0)
    # Trick by Emanuel: add a baseline STAT expression level. Otherwise, we
    # get no consumption of IL-2 for weak antigens and IL-2 remains high
    b = base / (1. - base)# Baseline = b/(1+b),  + (other stuff)/(1+b)
    coef = mm_r / (1. + b)
    return coef*i2 / (i2 + ec50hi*mm_r + ec50lo*(1. - mm_r)) + base


def deriv_r_parts(r, i2, p):
    """ p is a dict containing at least the following parameters:
        ho_rs, rho_r0, k_r, delta_r, delta_mr, K_js, n_js, ec50lo, ec50hi, r0
    """
    n = p["n_js"]
    jk = jak(r, i2, p["ec50lo"], p["ec50hi"], p["r0"], p["stat_baseline"])
    mrna_term = p["rho_r0"] + p["rho_rs"]*jk**n/(jk**n + p["K_js"]**n)
    # Plot the ratio production/degradation in log scale
    return [p["k_r"]/p["delta_mr"]*mrna_term, -p["delta_r"]*r]


def deriv_il2model(t, y, p):
    """ Derivative of the three dynamical variables of the Voisinne 2015 model
    when quasi-static approximation is applied to mRNA equations and
    the part about cyclinD and PI3K is ignored.

    Accomodate mixtures of antigens: parameter l_fracs lets us compute
    the number of ligands of each kind based on the total number in y.

    y = [Ag, IL-2R, IL-2]
    """
    d = np.zeros(y.shape)
    # Ag dynamics
    d[0] = -p["delta_a"]*p["N"]*p["Nstar"]/(p["N"] + p["Nstar"])*y[0]  # Ag

    # Compute the number of ligands of each type, if applicable.
    lig_nums = p["l_fracs"] * y[0]
    # Compute C_N with current Ag
    if isinstance(lig_nums, float):
        # p["tcr_tau"] should be a scalar
        # ratesp, tau1p, L1p, ri_tots, nmf
        complexes = steady_akpr_i_1ligand(p["tcr_rates"], p["tcr_tau"],
                        lig_nums, p["tcr_RItot"], p["tcr_nmf"])
        cn = complexes[p["tcr_nmf"][0]]
    elif len(lig_nums) == 2:
        # ratesp, tau1p, L1p, ri_tots, nmf
        # p["tcr_tau"] should be an array
        try:
            complexes = steady_akpr_i_2ligands(p["tcr_rates"], p["tcr_tau"],
                        lig_nums, p["tcr_RItot"], p["tcr_nmf"])
        except AssertionError as e:
            print(t, y)
            raise e
        n = p["tcr_nmf"][0]
        cn = complexes[n] + complexes[2*n+1]
    else:
        print("l_fracs was {}".format(p["l_fracs"]))
        raise ValueError("We can only have 1 or 2 ligand types for now.")

    # Normalize the C_N output
    # This is the tricky part: minimize the number of normalizations,
    # but also normalize differently for IL-2 production and IL-2R expression
    # Solution: allow two sets of thresholds on absolute cn values
    pon_cn_i2 = activation_function(cn, p["K_cni2"], pwr=p["n_ci2"])
    pon_cn_r = activation_function(cn, p["K_cnr"], pwr=p["n_cr"])

    # Compute other quasi-static variables
    jk = jak(y[1], y[2], p["ec50lo"], p["ec50hi"], p["r0"], p["stat_baseline"])
    pon_j_s = jk**p["n_js"] / (jk**p["n_js"] + p["K_js"]**p["n_js"])  # STAT phospho by JAK, which activates IL-2R
    # We can delete the inhibition of STAT by TCR and get very similar results
    if p["tcr_repress_stat"]:
        pon_j_s = pon_j_s * (1.0 - pon_cn_i2)

    # IL-2R
    #expo_factor = (1. - np.exp(-p["delta_mr"]*t))
    mrna_terms = p["rho_r0"] + p["rho_rc"]*pon_cn_r + p["rho_rs"]*pon_j_s
    d[1] = p["k_r"]/p["delta_mr"]*mrna_terms - p["delta_r"]*y[1]

    # IL-2 (in pM)
    ## TODO: check if we can remove the bounded exponential
    ## TODO: make N a dynamical variable instead of parameter.
    ## Would then need to include PI3K and cyclinD.
    expo_factor = (1. - np.exp(-p["delta_mi2"]*t))
    d[2] = expo_factor*p["ksec_i2"]*p["N"]*pon_cn_i2
    #d[2] = p["ksec_i2"]*p["N"]*pon_cn_i2
    d[2] = d[2] - p["kon_i2"]*p["N"]*y[1]*y[2]  # consumption by IL-2R
    # Decay: helps to get 0 for non-agonists if n_ci2 > 1
    # d[2] = d[2] - p["delta_i2"]*y[2]
    return d



def deriv_il2model_agdep(t, y, p):
    """ Derivative of the three dynamical variables of the Voisinne 2015 model
    when quasi-static approximation is applied to mRNA equations and
    the part about cyclinD and PI3K is ignored.

    Accomodate mixtures of antigens that may decay at different rates,
    depending on the number of receptors bound to them
    (i.e. on their qualities).

    y = [Ag1, Ag2, IL-2R, IL-2]
    """
    d = np.zeros(y.shape)
    # Compute CN with current Ag numbers
    delta_n_nstar = p["delta_a"]*p["N"]*p["Nstar"]/(p["N"] + p["Nstar"])
    if p["n_ligands"] == 1:
        # p["tcr_tau"] should be a scalar
        # ratesp, tau1p, L1p, ri_tots, nmf
        complexes = steady_akpr_i_1ligand(p["tcr_rates"], p["tcr_tau"],
                        y[0], p["tcr_RItot"], p["tcr_nmf"])
        cn = complexes[p["tcr_nmf"][0]]
        ctot = np.sum(complexes[0:p["tcr_nmf"][0]+1])
        d[0] = -delta_n_nstar*complexes[p["tcr_nmf"][1]]  # Ag
    elif p["n_ligands"] == 2:
        # ratesp, tau1p, L1p, ri_tots, nmf
        # p["tcr_tau"] should be an array
        try:
            complexes = steady_akpr_i_2ligands(p["tcr_rates"], p["tcr_tau"],
                        y[0:2], p["tcr_RItot"], p["tcr_nmf"])
        except AssertionError as e:
            print(t, y)
            raise e
        n = p["tcr_nmf"][0]
        cn = complexes[n] + complexes[2*n+1]   # total output
        ctot = np.sum(complexes[0:n+1])  # total C
        dtot = np.sum(complexes[n+1:2*n+2])  # total D
        d[0] = -delta_n_nstar * complexes[p["tcr_nmf"][1]]
        d[1] = -delta_n_nstar * complexes[p["tcr_nmf"][1] + n+1]
    else:
        print("n_ligands was {}".format(p["n_ligands"]))
        raise ValueError("We can only have 1 or 2 ligand types for now.")

    # Normalize the C_N output
    # This is the tricky part: minimize the number of normalizations,
    # but also normalize differently for IL-2 production and IL-2R expression
    # Solution: allow two sets of thresholds on absolute cn values
    pon_cn_i2 = activation_function(cn, p["K_cni2"], pwr=p["n_ci2"])
    pon_cn_r = activation_function(cn, p["K_cnr"], pwr=p["n_cr"])

    # Compute other quasi-static variables
    i_il2r = p["n_ligands"]
    i_il2 = i_il2r + 1
    jk = jak(y[i_il2r], y[i_il2], p["ec50lo"], p["ec50hi"], p["r0"], p["stat_baseline"])
    pon_j_s = jk**p["n_js"] / (jk**p["n_js"] + p["K_js"]**p["n_js"])  # STAT phospho by JAK, which activates IL-2R
    # We can delete the inhibition of STAT by TCR and get very similar results
    if p["tcr_repress_stat"]:
        pon_j_s = pon_j_s * (1.0 - pon_cn_i2)

    # IL-2R
    #expo_factor = (1. - np.exp(-p["delta_mr"]*t))
    mrna_terms = p["rho_r0"] + p["rho_rc"]*pon_cn_r + p["rho_rs"]*pon_j_s
    d[i_il2r] = p["k_r"]/p["delta_mr"]*mrna_terms - p["delta_r"]*y[i_il2r]

    # IL-2 (in pM)
    ## TODO: check if we can remove the bounded exponential
    ## TODO: make N a dynamical variable instead of parameter.
    ## Would then need to include PI3K and cyclinD.
    expo_factor = (1. - np.exp(-p["delta_mi2"]*t))
    d[i_il2] = expo_factor*p["ksec_i2"]*p["N"]*pon_cn_i2
    #d[2] = p["ksec_i2"]*p["N"]*pon_cn_i2
    d[i_il2] = d[i_il2] - p["kon_i2"]*p["N"]*y[i_il2]*y[i_il2r]  # consumption by IL-2R
    # Decay: helps to get 0 for non-agonists if n_ci2 > 1
    # d[2] = d[2] - p["delta_i2"]*y[2]
    return d
