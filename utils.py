import os
from scipy.interpolate import interp1d
from scipy.stats import binned_statistic
import numpy as np
import h5py
import pandas as pd


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

    if bins == None:
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
    func = lambda y: weighted_quantile(y, 0.5, sample_weight=w)
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


def get_reg_data(ii, tag, data_fields, inp='FLARES'):
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

    with h5py.File(sim, 'r') as hf:
        s_len = hf[tag + '/Galaxy'].get('S_Length')
        if s_len is not None:
            for f in data_fields:
                f_splt = f.split(",")

                # Extract this dataset
                if len(f_splt) > 1:
                    key = tag + '/' + f_splt[0]
                    d = np.array(hf[key].get(f_splt[1]))

                    # If it is multidimensional it needs transposing
                    if len(d.shape) > 1:
                        data[f] = d.T
                    else:
                        data[f] = d

        else:

            for f in data_fields:
                data[f] = np.array([])

    return data


def get_data(sim, regions, snap, data_fields, length_key="Galaxy,S_Length"):
    # Load weights
    df = pd.read_csv('../weight_files/weights_grid.txt')
    weights = np.array(df['weights'])

    # Initialise dictionary to store results
    data = {k: [] for k in data_fields}
    data["weights"] = []
    data["begin"] = []
    data["gbegin"] = []

    # Initialise particle offsets
    offset = 0

    # Loop over regions and snapshots
    for reg in regions:
        reg_data = get_reg_data(reg, snap, data_fields, inp=sim)

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
            data["weights"].extend(np.full(reg_data[length_key].size,
                                           weights[int(reg)]))
        else:
            data["weights"].extend(np.ones(len(reg_data[length_key])))

        # Add on new offset
        offset = len(data[data_fields[0]])

    # Convert lists to arrays
    for key in data:
        data[key] = np.array(data[key])

    return data

