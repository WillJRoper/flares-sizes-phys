import os
from scipy.interpolate import interp1d
import numpy as np


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def calc_3drad(poss):
    # Get galaxy particle indices
    rs = np.sqrt(poss[:, 0] ** 2 + poss[:, 1] ** 2 + poss[:, 2] ** 2)

    return rs


def calc_light_mass_rad(rs, ls, radii_frac=0.5):

    if ls.size < 10:
        return 0.0

    # Sort the radii and masses
    sinds = np.argsort(rs)
    rs = rs[sinds]
    ls = ls[sinds]

    if ls.size < 10:
        return 0.0

    # Get the cumalative sum of masses
    l_profile = np.cumsum(ls)

    # Get the total mass and half the total mass
    tot_l = np.sum(ls)
    half_l = tot_l * radii_frac

    # Get the half mass radius particle
    hmr_ind = np.argmin(np.abs(l_profile - half_l))
    l_profile_cutout = l_profile[np.max((hmr_ind - 10, 0)):
                                 np.min((hmr_ind + 10, l_profile.size))]
    rs_cutout = rs[np.max((hmr_ind - 10, 0)):
                   np.min((hmr_ind + 10, l_profile.size))]

    if len(rs_cutout) < 3:
        return 0

    # Interpolate the arrays for better resolution
    interp_func = interp1d(rs_cutout, l_profile_cutout, kind="linear")
    interp_rs = np.linspace(rs_cutout.min(), rs_cutout.max(), 500)
    interp_1d_ls = interp_func(interp_rs)

    new_hmr_ind = np.argmin(np.abs(interp_1d_ls - half_l))
    hmr = interp_rs[new_hmr_ind]

    return hmr