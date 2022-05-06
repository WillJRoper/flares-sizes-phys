import numpy as np
import matplotlib.pyplot as plt
from flare import plt as flareplt
from utils import mkdir, plot_meidan_stat, age2z

import eagle_IO.eagle_IO as eagle_io


def plot_birth_met(stellar_data, snap, weight_norm, path):

    # Define redshift
    z = float(snap.split("z")[-1].replace("p", "."))

    # Get the index for this star particle
    s_inds = stellar_data["Particle,S_Index"]

    # Get the region for each galaxy
    regions = stellar_data["regions"]

    # Get the arrays from the raw data files
    aborn = eagle_io.read_array('PARTDATA', path.replace("<reg>", "00"), snap,
                                'PartType4/StellarFormationTime',
                                numThreads=8)

    # Extract arrays
    zs = np.zeros(s_inds.size)
    mets = stellar_data["Particle,S_Z_smooth"]
    w = np.zeros(s_inds.size)

    # Extract weights for each particle
    prev_reg = 0
    for igal in range(stellar_data["begin"].size):

        # Extract galaxy range
        b = stellar_data["begin"][igal]
        e = b + stellar_data["Galaxy,S_Length"][igal]
        this_s_inds = s_inds[b: e]

        # Get this galaxies region
        reg = regions[igal]

        # Set weights for these particles
        w[b: e] = stellar_data["weights"][igal]

        # Open a new region file if necessary
        if reg != prev_reg:
            # Get the arrays from the raw data files
            aborn = eagle_io.read_array('PARTDATA',
                                        path.replace("<reg>",
                                                     str(reg).zfill(2)),
                                        snap,
                                        'PartType4/StellarFormationTime',
                                        numThreads=8)
            prev_reg = reg

        # Get this galaxies data
        zs[b: e] = 1 / aborn[this_s_inds] - 1

    # Set up the plot
    fig = plt.figure(figsize=(3.5, 3.5))
    ax = fig.add_subplot(111)

    # Remove anomalous values
    okinds = np.logical_and(zs > 0, mets > 0)

    im = ax.hexbin(zs[okinds], mets[okinds], gridsize=50,
                   mincnt=np.min(w) - (0.1 * np.min(w)),
                   C=w[okinds],
                   extent=[4.5, 15, -5, 0],
                   reduce_C_function=np.sum, yscale='log',
                   norm=weight_norm, linewidths=0.2,
                   cmap='viridis')

    ax.set_ylabel(r"$Z_{\mathrm{birth}}$")
    ax.set_xlabel(r"$z_{\mathrm{birth}}$")

    # Save figure
    mkdir("plots/stellar_evo/")
    fig.savefig("plots/stellar_evo/stellar_birthZ_%s.png" % snap,
                bbox_inches="tight")


def plot_birth_den(stellar_data, snap, weight_norm, path):

    # Define redshift
    z = float(snap.split("z")[-1].replace("p", "."))

    # Get the index for this star particle
    s_inds = stellar_data["Particle,S_Index"]

    # Get the region for each galaxy
    regions = stellar_data["regions"]

    # Get the arrays from the raw data files
    aborn = eagle_io.read_array('PARTDATA', path.replace("<reg>", "00"), snap,
                                'PartType4/StellarFormationTime',
                                numThreads=8)
    den_born = eagle_io.read_array("PARTDATA", path.replace("<reg>", "00"),
                                   snap, "PartType4/BirthDensity",
                                   numthreads=8)

    # Extract arrays
    zs = np.zeros(s_inds.size)
    dens = np.zeros(s_inds.size)
    w = np.zeros(s_inds.size)

    # Extract weights for each particle
    prev_reg = 0
    for igal in range(stellar_data["begin"].size):

        # Extract galaxy range
        b = stellar_data["begin"][igal]
        e = b + stellar_data["Galaxy,S_Length"][igal]
        this_s_inds = s_inds[b: e]

        # Get this galaxies region
        reg = regions[igal]

        # Set weights for these particles
        w[b: e] = stellar_data["weights"][igal]

        # Open a new region file if necessary
        if reg != prev_reg:
            # Get the arrays from the raw data files
            aborn = eagle_io.read_array('PARTDATA',
                                        path.replace("<reg>",
                                                     str(reg).zfill(2)),
                                        snap,
                                        'PartType4/StellarFormationTime',
                                        numThreads=8)
            den_born = eagle_io.read_array("PARTDATA",
                                           path.replace("<reg>",
                                                        str(reg).zfill(2)),
                                           snap, "PartType4/BirthDensity",
                                           numthreads=8)
            prev_reg = reg

        # Get this galaxies data
        zs[b: e] = 1 / aborn[this_s_inds] - 1
        dens[b: e] = den_born[this_s_inds]

    # Set up the plot
    fig = plt.figure(figsize=(3.5, 3.5))
    ax = fig.add_subplot(111)

    # Remove anomalous values
    okinds = np.logical_and(zs > 0, dens > 0)

    im = ax.hexbin(zs[okinds], dens[okinds], gridsize=50,
                   mincnt=np.min(w) - (0.1 * np.min(w)),
                   C=w[okinds],
                   reduce_C_function=np.sum, yscale='log',
                   norm=weight_norm, linewidths=0.2,
                   cmap='viridis')

    ax.set_ylabel(r"$\rho_{\mathrm{birth}} / $ ???")
    ax.set_xlabel(r"$z_{\mathrm{birth}}$")

    # Save figure
    mkdir("plots/stellar_evo/")
    fig.savefig("plots/stellar_evo/stellar_birthden_%s.png" % snap,
                bbox_inches="tight")
