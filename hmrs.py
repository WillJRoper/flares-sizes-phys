import os
from pathlib import Path

import h5py
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from utils import calc_3drad, calc_light_mass_rad

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


def get_reg_data(ii, tag, data_fields, inp='FLARES', offset=0, goffset=0):
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

            data["begin"] = np.zeros(len(data["Galaxy,S_Length"]),
                                     dtype=np.int64) + offset
            data["begin"][1:] = np.cumsum(data["Galaxy,S_Length"])[:-1]

            data["gbegin"] = np.zeros(len(data["Galaxy,G_Length"]),
                                      dtype=np.int64) + goffset
            data["gbegin"][1:] = np.cumsum(data["Galaxy,G_Length"])[:-1]

        else:

            for f in data_fields:
                data[f] = np.array([])

    return data


def get_data(sim, regions, snap, data_fields):

    # Load weights
    df = pd.read_csv('../weight_files/weights_grid.txt')
    weights = np.array(df['weights'])

    # Initialise dictionary to store results
    data = {k: [] for k in data_fields}
    data["weights"] = []

    # Initialise particle offsets
    offset = 0
    goffset = 0

    # Loop over regions and snapshots
    for reg in regions:
        reg_data = get_reg_data(reg, snap, data_fields, inp=sim,
                                offset=offset, goffset=goffset)

        # Combine this region
        for f in data_fields:
            data[f].extend(reg_data[f])

        # Include this regions weighting
        if sim == "FLARES":
            data["weights"].extend(np.full(reg_data["begin"].size,
                                           weights[int(reg)]))
        else:
            data["weights"].extend(np.ones(len(reg_data["begin"])))

        # Add on new offset
        offset = len(data[data_fields[0]])
        goffset = len(data[data_fields[1]])
        print(offset, len(data["weights"]))

    # Convert lists to arrays
    for key in data:
        data[key] = np.array(data[key])

    return data


def plot_stellar_hmr(sim, regions, snap, weight_norm):

    # Define data fields
    data_fields = ("Particle,S_Mass", "Particle,G_Mass",
                   "Particle,S_Coordinates", "Particle,G_Coordinates",
                   "Particle/Apertures/Star,30", "Particle/Apertures/Gas,30",
                   "Galaxy,COP", "Galaxy,S_Length", "Galaxy,G_Length",
                   "Galaxy,GroupNumber", "Galaxy,SubGroupNumber",
                   "begin", "gbegin",)

    # Get the data
    data = get_data(sim, regions, snap, data_fields)

    # Define arrays to store computations
    hmrs = np.zeros(len(data["begin"]))
    mass = np.zeros(len(data["begin"]))
    w = np.zeros(len(data["begin"]))

    # Loop over galaxies and calculate stellar HMR
    for (igal, b), l in zip(enumerate(data["begin"]), data["Galaxy,S_Length"]):

        # Get this galaxy's data
        app = data["Particle/Apertures/Star,30"][b: l]
        cop = data["Galaxy,COP"][igal]
        ms = data["Particle,S_Mass"][b: l][app]

        # Compute particle radii
        rs = calc_3drad(data["Particle,S_Coordinates"][b: l, :] - cop)

        # Compute HMR
        hmr = calc_light_mass_rad(rs, ms, radii_frac=0.5)

        # Store results
        mass[igal] = np.sum(ms) * 10 ** 10
        hmrs[igal] = hmr
        w[igal] = data["weights"][igal]

    # Set up plot
    fig = plt.figure(figsize=(3.5, 3.5))
    ax = fig.add_subplot(111)
    ax.loglog()

    # Plot data
    ax.hexbin(mass, hmrs, gridsize=50,
              mincnt=np.min(w) - (0.1 * np.min(w)),
              C=w,
              reduce_C_function=np.sum, xscale='log', yscale='log',
              norm=weight_norm, linewidths=0.2, cmap='viridis')

    # Label axes
    ax.set_xlabel("$M_\star / M_\odot$")
    ax.set_ylabel("$R_{1/2} / [\mathrm{pkpc}]$")

    # Save figure
    Path("/plots/stellar_hmr/").mkdir(parents=True, exist_ok=True)
    fig.savefig("/plots/stellar_hmr/stellar_hmr_%s.png")
