import os

import h5py
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils import calc_3drad, calc_light_mass_rad, mkdir, plot_meidan_stat

os.environ['FLARE'] = '/cosma7/data/dp004/dc-wilk2/flare'
mpl.use('Agg')

from flare import plt as flareplt

# Set plotting fontsizes
plt.rcParams['axes.grid'] = True
flareplt.rcParams['axes.grid'] = True

SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


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


def plot_stellar_density(sim, regions, snap, weight_norm):
    # Define x and y limits
    hmrlims = (-3.5, 1)
    mlims = (7, 11)
    denlims = (2, 17.5)

    # Define data fields
    stellar_data_fields = ("Particle,S_Mass", "Particle,S_Coordinates",
                           "Particle/Apertures/Star,1",
                           "Particle/Apertures/Star,5",
                           "Particle/Apertures/Star,10",
                           "Particle/Apertures/Star,30", "Galaxy,COP",
                           "Galaxy,S_Length", "Galaxy,GroupNumber",
                           "Galaxy,SubGroupNumber")

    # Get the data
    stellar_data = get_data(sim, regions, snap, stellar_data_fields,
                            length_key="Galaxy,S_Length")

    # Define arrays to store computations
    hmrs = np.zeros(len(stellar_data["begin"]))
    mass = np.zeros(len(stellar_data["begin"]))
    den = {key: np.zeros(len(stellar_data["begin"]))
           for key in [0.05, 0.1, 0.5, 1, 5, 10, 30]}
    den_hmr = np.zeros(len(stellar_data["begin"]))
    w = np.zeros(len(stellar_data["begin"]))

    # Loop over galaxies and calculate stellar HMR and denisty within HMR
    for (igal, b), l in zip(enumerate(stellar_data["begin"]),
                            stellar_data["Galaxy,S_Length"]):

        if l < 100:
            continue

        # Get this galaxy's stellar_data
        app = stellar_data["Particle/Apertures/Star,30"][b: b + l]
        cop = stellar_data["Galaxy,COP"][igal]
        ms = stellar_data["Particle,S_Mass"][b: b + l][app]
        pos = stellar_data["Particle,S_Coordinates"][b: b + l, :][app]

        # Compute particle radii
        rs = calc_3drad(pos - cop)

        # Compute HMR
        hmr = calc_light_mass_rad(rs, ms, radii_frac=0.5)

        # Store results
        den_hmr[igal] = (np.sum(ms[rs <= hmr]) * 10 ** 10
                         / (4 / 3 * np.pi * hmr ** 3))
        hmrs[igal] = hmr
        mass[igal] = np.sum(ms) * 10 ** 10
        w[igal] = stellar_data["weights"][igal]

        for r in [0.05, 0.1, 0.5]:
            den[r][igal] = (np.sum(ms[rs <= r]) * 10 ** 10
                            / (4 / 3 * np.pi * r ** 3))

    # Loop over galaxies and calculate denisty within radii
    for (igal, b), l in zip(enumerate(stellar_data["begin"]),
                            stellar_data["Galaxy,S_Length"]):

        for r in den:

            if r in [0.05, 0.1, 0.5]:
                continue

            # Get this galaxy's stellar_data for this radii
            app = stellar_data["Particle/Apertures/Star,%d" % r][b: b + l]
            ms = stellar_data["Particle,S_Mass"][b: b + l][app]

            # Compute density
            den[r][igal] = np.sum(ms) * 10 ** 10 / (4 / 3 * np.pi * r ** 3)

    # Remove galaxies without stars
    okinds = np.logical_and(den_hmr > 0, hmrs > 0)
    print("Galaxies before spurious cut: %d" % den_hmr.size)
    den_hmr = den_hmr[okinds]
    hmrs = hmrs[okinds]
    mass = mass[okinds]
    w = w[okinds]
    for r in den:
        den[r] = den[r][okinds]
    print("Galaxies after spurious cut: %d" % den_hmr.size)

    # Define how mnay columns
    ncols = 1 + len(den)

    # Set up plot
    fig = plt.figure(figsize=(2.25 * ncols, 2.25))
    gs = gridspec.GridSpec(nrows=1, ncols=ncols + 1,
                           width_ratios=[20, ] * ncols + [1])
    gs.update(wspace=0.0, hspace=0.0)
    axes = []
    cax = fig.add_subplot(gs[:, -1])
    i = 0
    while i < ncols:
        axes.append(fig.add_subplot(gs[0, i]))
        if i > 0:
            axes[i].loglog()
            axes[i].tick_params(axis='y', left=False, right=False,
                                labelleft=False, labelright=False)
        i += 1

    # Plot stellar_data
    im = axes[0].hexbin(hmrs, den_hmr, gridsize=50,
                        mincnt=np.min(w) - (0.1 * np.min(w)),
                        C=w,
                        extent=[hmrlims[0], hmrlims[1], denlims[0],
                                denlims[1]],
                        reduce_C_function=np.sum, xscale='log', yscale='log',
                        norm=weight_norm, linewidths=0.2, cmap='viridis')

    # Plot weighted medians
    for i, r in enumerate(den):
        okinds = np.logical_and(den[r] > 0, hmrs > 0)
        axes[i].hexbin(hmrs[okinds], den[r][okinds], gridsize=50,
                       mincnt=np.min(w[okinds]) - (0.1 * np.min(w)),
                       C=w[okinds],
                       extent=[hmrlims[0], hmrlims[1], denlims[0], denlims[1]],
                       reduce_C_function=np.sum, xscale='log', yscale='log',
                       norm=weight_norm, linewidths=0.2, cmap='viridis')
        p = plot_meidan_stat(hmrs[okinds], den[r][okinds], w[okinds],
                             axes[i + 1], "R=%.2f" % r,
                             color=None, bins=None, ls='--')

    p = plot_meidan_stat(hmrs, den_hmr, w, axes[0], "$R=R_{1/2}$",
                         color=None, bins=None, ls='-')

    # Set lims
    for ax in axes:
        ax.set_ylim(10 ** denlims[0], 10 ** denlims[1])
        ax.set_xlim(10 ** hmrlims[0], 10 ** hmrlims[1])

    # Set titles
    axes[0].set_title("$R=R_{1/2}$")
    for i, r in enumerate(den):
        axes[i + 1].set_title("R=%.2f" % r)

    # Label axes
    axes[0].set_ylabel(r"$\rho_\star(<R) / [M_\odot / \mathrm{pkpc}^3]$")
    for i in range(ncols):
        axes[i].set_xlabel("$R_{1/2} / [\mathrm{pkpc}]$")

    cbar = fig.colorbar(im, cax)
    cbar.set_label("$\sum w_{i}$")

    # Save figure
    mkdir("plots/density/")
    fig.savefig("plots/density/stellar_density_hmr_%s.png" % snap,
                bbox_inches="tight")

    plt.close(fig)

    # Set up plot
    fig = plt.figure(figsize=(2.25 * ncols, 2.25))
    gs = gridspec.GridSpec(nrows=1, ncols=ncols + 1,
                           width_ratios=[20, ] * ncols + [1])
    gs.update(wspace=0.0, hspace=0.0)
    axes = []
    cax = fig.add_subplot(gs[:, -1])
    i = 0
    while i < ncols:
        axes.append(fig.add_subplot(gs[0, i]))
        if i > 0:
            axes[i].loglog()
            axes[i].tick_params(axis='y', left=False, right=False,
                                labelleft=False, labelright=False)
        i += 1

    # Plot stellar_data
    im = axes[0].hexbin(mass, den_hmr, gridsize=50,
                        mincnt=np.min(w) - (0.1 * np.min(w)),
                        C=w,
                        extent=[mlims[0], mlims[1], denlims[0], denlims[1]],
                        reduce_C_function=np.sum, xscale='log', yscale='log',
                        norm=weight_norm, linewidths=0.2, cmap='viridis')

    # Plot weighted medians
    for i, r in enumerate(den):
        okinds = np.logical_and(den[r] > 0, mass > 0)
        axes[i].hexbin(mass[okinds], den[r][okinds], gridsize=50,
                       mincnt=np.min(w[okinds]) - (0.1 * np.min(w)),
                       C=w[okinds],
                       extent=[mlims[0], mlims[1], denlims[0], denlims[1]],
                       reduce_C_function=np.sum, xscale='log', yscale='log',
                       norm=weight_norm, linewidths=0.2, cmap='viridis')
        p = plot_meidan_stat(mass[okinds], den[r][okinds], w[okinds],
                             axes[i + 1], "R=%.2f" % r,
                             color=None, bins=None, ls='--')

    p = plot_meidan_stat(mass, den_hmr, w, axes[0], "$R=R_{1/2}$",
                         color=None, bins=None, ls='-')

    # Set lims
    for ax in axes:
        ax.set_ylim(10 ** denlims[0], 10 ** denlims[1])
        ax.set_xlim(10 ** mlims[0], 10 ** mlims[1])

    # Set titles
    axes[0].set_title("$R=R_{1/2}$")
    for i, r in enumerate(den):
        axes[i + 1].set_title("R=%.2f" % r)

    # Label axes
    axes[0].set_ylabel(r"$\rho_\star(<R) / [M_\odot / \mathrm{pkpc}^3]$")
    for i in range(ncols):
        axes[i].set_xlabel("$M_\star / M_\odot$")

    cbar = fig.colorbar(im, cax)
    cbar.set_label("$\sum w_{i}$")

    # Save figure
    mkdir("plots/density/")
    fig.savefig("plots/density/stellar_density_mass_%s.png" % snap,
                bbox_inches="tight")

    plt.close(fig)
