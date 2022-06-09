import os
from scipy.interpolate import interp1d
from scipy.stats import binned_statistic
import numpy as np
import h5py
import pandas as pd
from astropy.cosmology import Planck18 as cosmo, z_at_value
import astropy.units as u


def calc_ages(z, a_born):
    # Convert scale factor into redshift
    z_born = 1 / a_born - 1

    # Convert to time in Gyrs
    t = cosmo.age(z)
    t_born = cosmo.age(z_born)

    # Calculate the VR
    ages = (t - t_born).to(u.Myr)

    return ages.value


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def calc_3drad(poss):
    # Get galaxy particle indices
    rs = np.sqrt(poss[:, 0] ** 2 + poss[:, 1] ** 2 + poss[:, 2] ** 2)

    return rs


def age2z(age, z):

    # Apply units to age
    age *= u.Gyr

    # Define Universe age in Gyrs
    current_age = cosmo.age(z)

    # Universe at which star was born
    birth_age = current_age - age

    # Compute redshift of birth_age
    birth_z = z_at_value(cosmo.age, birth_age, zmin=0, zmax=50)

    return birth_z


def calc_light_mass_rad(rs, ls, radii_frac=0.5):

    if ls.size < 10:
        return 0.0

    # Sort the radii and masses
    sinds = np.argsort(rs)
    rs = rs[sinds]
    ls = ls[sinds]

    # Get the cumalative sum of masses
    l_profile = np.cumsum(ls)

    # Get the total mass and half the total mass
    tot_l = np.sum(ls)
    half_l = tot_l * radii_frac

    # Get the half mass radius particle
    hmr_ind = np.argmin(np.abs(l_profile - half_l))
    # l_profile_cutout = l_profile[np.max((hmr_ind - 10, 0)):
    #                              np.min((hmr_ind + 10, l_profile.size))]
    # rs_cutout = rs[np.max((hmr_ind - 10, 0)):
    #                np.min((hmr_ind + 10, l_profile.size))]
    #
    # if len(rs_cutout) < 3:
    #     return 0
    #
    # # Interpolate the arrays for better resolution
    # interp_func = interp1d(rs_cutout, l_profile_cutout, kind="linear")
    # interp_rs = np.linspace(rs_cutout.min(), rs_cutout.max(), 500)
    # interp_1d_ls = interp_func(interp_rs)
    #
    # new_hmr_ind = np.argmin(np.abs(interp_1d_ls - half_l))
    # hmr = interp_rs[new_hmr_ind]

    return rs[hmr_ind]


def plot_meidan_stat(xs, ys, w, ax, lab, color, bins=None, ls='-'):

    if bins is None:
        bin = np.logspace(np.log10(xs.min()), np.log10(xs.max()), 20)
    else:
        zs = np.float64(xs)

        uniz = np.unique(zs)
        bin_wids = uniz[1:] - uniz[:-1]
        low_bins = uniz[:-1] - (bin_wids / 2)
        high_bins = uniz[:-1] + (bin_wids / 2)
        low_bins = list(low_bins)
        high_bins = list(high_bins)
        low_bins.append(high_bins[-1])
        high_bins.append(uniz[-1] + 1)
        low_bins = np.array(low_bins)
        high_bins = np.array(high_bins)

        bin = np.zeros(uniz.size + 1)
        bin[:-1] = low_bins
        bin[1:] = high_bins

    # Compute binned statistics
    def func(y):
        return weighted_quantile(y, 0.5, sample_weight=w)
    y_stat, binedges, bin_ind = binned_statistic(xs, ys,
                                                 statistic=func, bins=bin)

    # Compute bincentres
    bin_wid = binedges[1] - binedges[0]
    bin_cents = binedges[1:] - bin_wid / 2

    okinds = np.logical_and(~np.isnan(bin_cents), ~np.isnan(y_stat))

    if color is not None:
        return ax.plot(bin_cents[okinds], y_stat[okinds], color=color,
                       linestyle=ls, label=lab)
    else:
        return ax.plot(bin_cents[okinds], y_stat[okinds], color=color,
                       linestyle=ls, label=lab)


def weighted_quantile(values, quantiles, sample_weight=None,
                      values_sorted=False, old_style=False):
    """
    Taken from From https://stackoverflow.com/a/29677616/1718096
    Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """

    # do some housekeeping
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), \
        'quantiles should be in [0, 1]'

    # if not sorted, sort values array
    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)


def get_reg_data(ii, tag, data_fields, inp='FLARES', length_key="Galaxy,S_Length"):
    num = str(ii)
    if inp == 'FLARES':
        if len(num) == 1:
            num = '0' + num

        sim = rF"/cosma7/data/dp004/dc-payy1/my_files/flares_pipeline/data/" \
              rF"FLARES_{num}_sp_info.hdf5"

    else:
        sim = rF"/cosma7/data/dp004/dc-payy1/my_files/flares_pipeline/data/" \
              rF"EAGLE_{inp}_sp_info.hdf5"

    # Initialise dictionary to store data
    data = {}

    # Get redshift
    z_str = tag.split('z')[1].split('p')
    z = float(z_str[0] + '.' + z_str[1])

    with h5py.File(sim, 'r') as hf:
        splt_len_key = length_key.split(",")
        s_len = hf[tag + "/" + splt_len_key[0]].get(splt_len_key[1])
        if s_len is not None:
            for f in data_fields:
                f_splt = f.split(",")

                # Extract this dataset
                if len(f_splt) > 1:
                    key = tag + '/' + f_splt[0]
                    d = np.array(hf[key].get(f_splt[1]))

                    # Apply conversion from cMpc to pkpc
                    if "Coordinates" in f_splt[1] or "COP" in f_splt[1]:
                        d *= (1 / (1 + z) * 10**3)

                    # If it is multidimensional it needs transposing
                    if len(d.shape) > 1:
                        data[f] = d.T
                    else:
                        data[f] = d

        else:

            for f in data_fields:
                data[f] = np.array([])

    return data


def get_snap_data(sim, regions, snap, data_fields,
                  length_key="Galaxy,S_Length"):

    # Load weights
    df = pd.read_csv('../weight_files/weights_grid.txt')
    weights = np.array(df['weights'])

    # Initialise dictionary to store results
    data = {k: [] for k in data_fields}
    data["weights"] = []
    data["regions"] = []
    data["begin"] = []

    # Initialise particle offsets
    offset = 0

    # Loop over regions and snapshots
    for reg in regions:
        reg_data = get_reg_data(reg, snap, data_fields,
                                inp=sim, length_key=length_key)

        # Combine this region
        for f in data_fields:
            data[f].extend(reg_data[f])

        # Define galaxy start index arrays
        start_index = np.full(reg_data[length_key].size,
                              offset, dtype=int)
        start_index[1:] += np.cumsum(reg_data[length_key][:-1])
        data["begin"].extend(start_index)

        # Include this regions weighting
        if sim == "FLARES":
            data["regions"].extend(np.full(reg_data[length_key].size,
                                           int(reg)))
            data["weights"].extend(np.full(reg_data[length_key].size,
                                           weights[int(reg)]))
        else:
            data["regions"].extend(np.ones(len(reg_data[length_key])))
            data["weights"].extend(np.ones(len(reg_data[length_key])))

        # Add on new offset
        offset = len(data[data_fields[0]])

    # Convert lists to arrays
    for key in data:
        data[key] = np.array(data[key])

    return data


def clean_data(stellar_data, gas_data):

    # Get length array
    slen = stellar_data["Galaxy,S_Length"]
    n_gal = slen.size

    # Create boolean mask
    okinds = slen >= 100

    # Loop over keys and mask necessary arrays
    for key in stellar_data:

        # Read array
        arr = stellar_data[key]
        if arr.size == n_gal:
            arr = arr[okinds]

    # Loop over keys and mask necessary arrays
    for key in gas_data:

        # Read array
        arr = gas_data[key]
        if arr.size == n_gal:
            arr = arr[okinds]

    return stellar_data, gas_data
