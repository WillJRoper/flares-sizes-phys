import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from flare import plt as flareplt
from utils import mkdir, plot_meidan_stat, age2z
from unyt import mh, cm, Gyr, g, Msun, Mpc

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
                                noH=True, physicalUnits=True,
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
                                        snap, noH=True, physicalUnits=True,
                                        'PartType4/StellarFormationTime',
                                        numThreads=8)
            prev_reg = reg

        # Get this galaxies data
        zs[b: e] = 1 / aborn[this_s_inds] - 1

    # Set up the plot
    fig = plt.figure(figsize=(3.5, 3.5))
    ax = fig.add_subplot(111)

    # Remove anomalous values
    okinds = np.ones(mets.size, dtype=bool)

    im = ax.hexbin(zs[okinds], mets[okinds], gridsize=50,
                   mincnt=np.min(w) - (0.1 * np.min(w)),
                   C=w[okinds],
                   extent=[4.5, 15, 0, 1], norm=LogNorm(),
                   reduce_C_function=np.sum, linewidths=0.2,
                   cmap='viridis')

    ax.set_ylabel(r"$Z_{\mathrm{birth}}$")
    ax.set_xlabel(r"$z_{\mathrm{birth}}$")

    fig.colorbar(im)

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
                                noH=True, physicalUnits=True,
                                numThreads=8)
    den_born = (eagle_io.read_array("PARTDATA", path.replace("<reg>", "00"),
                                    snap, "PartType4/BirthDensity",
                                    noH=True, physicalUnits=True,
                                    numThreads=8) * 10**10
                * Msun / Mpc ** 3 / mh).to(1 / cm ** 3).value

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
                                        snap, noH=True, physicalUnits=True,
                                        'PartType4/StellarFormationTime',
                                        numThreads=8)
            den_born = (eagle_io.read_array("PARTDATA",
                                            path.replace("<reg>",
                                                         str(reg).zfill(2)),
                                            snap, "PartType4/BirthDensity",
                                            noH=True, physicalUnits=True,
                                            numThreads=8) * 10**10
                        * Msun / Mpc ** 3 / mh).to(1 / cm ** 3).value
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
                   C=w[okinds], norm=LogNorm(),
                   reduce_C_function=np.sum, yscale='log',
                   linewidths=0.2,
                   cmap='viridis')

    ax.set_ylabel(r"$\rho_{\mathrm{birth}} / $ \mathrm{cm}^{-3}")
    ax.set_xlabel(r"$z_{\mathrm{birth}}$")

    fig.colorbar(im)

    # Save figure
    mkdir("plots/stellar_evo/")
    fig.savefig("plots/stellar_evo/stellar_birthden_%s.png" % snap,
                bbox_inches="tight")


def plot_birth_den_vs_met(stellar_data, snap, weight_norm, path):

    # Define redshift bins
    zbins = list(range(5, 11, 2))
    zbins.append(np.inf)

    # Define EAGLE subgrid  parameters
    parameters = {"f_th,min": 0.3,
                  "f_th,max": 3,
                  "n_Z": 1.0,
                  "n_n": 1.0,
                  "Z_pivot": 0.1 * 0.012,
                  "n_pivot": 0.67}

    star_formation_parameters = {"threshold_Z0": 0.002,
                                 "threshold_n0": 0.1,
                                 "slope": -0.64}

    number_of_bins = 128

    # Constants; these could be put in the parameter file but are
    # rarely changed
    birth_density_bins = np.logspace(-3, 6.8, number_of_bins)
    metal_mass_fraction_bins = np.logspace(-5.9, 0, number_of_bins)

    # Now need to make background grid of f_th.
    birth_density_grid, metal_mass_fraction_grid = np.meshgrid(
        0.5 * (birth_density_bins[1:] + birth_density_bins[:-1]),
        0.5 * (metal_mass_fraction_bins[1:] + metal_mass_fraction_bins[:-1]))

    f_th_grid = parameters["f_th,min"] + (parameters["f_th,max"]
                                          - parameters["f_th,min"]) / (
        1.0
        + (metal_mass_fraction_grid /
           parameters["Z_pivot"]) ** parameters["n_Z"]
        * (birth_density_grid / parameters["n_pivot"]) ** (-parameters["n_n"])
    )

    axlims_x = []
    axlims_y = []

    # Define redshift
    z = float(snap.split("z")[-1].replace("p", "."))

    # Get the index for this star particle
    s_inds = stellar_data["Particle,S_Index"]

    # Get the region for each galaxy
    regions = stellar_data["regions"]

    # Get the arrays from the raw data files
    aborn = eagle_io.read_array('PARTDATA', path.replace("<reg>", "00"), snap,
                                'PartType4/StellarFormationTime',
                                noH=True, physicalUnits=True,
                                numThreads=8)
    den_born = (eagle_io.read_array("PARTDATA", path.replace("<reg>", "00"),
                                    snap, "PartType4/BirthDensity",
                                    noH=True, physicalUnits=True,
                                    numThreads=8) * 10**10
                * Msun / Mpc ** 3 / mh).to(1 / cm ** 3).value

    # Extract arrays
    zs = np.zeros(s_inds.size)
    dens = np.zeros(s_inds.size)
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
                                        snap, noH=True, physicalUnits=True,
                                        'PartType4/StellarFormationTime',
                                        numThreads=8)
            den_born = (eagle_io.read_array("PARTDATA",
                                            path.replace("<reg>",
                                                         str(reg).zfill(2)),
                                            snap, "PartType4/BirthDensity",
                                            noH=True, physicalUnits=True,
                                            numThreads=8) * 10**10
                        * Msun / Mpc ** 3 / mh).to(1 / cm ** 3).value
            prev_reg = reg

        # Get this galaxies data
        zs[b: e] = 1 / aborn[this_s_inds] - 1
        dens[b: e] = den_born[this_s_inds]

    # Set up the plot
    ncols = len(zbins) - 1

    # Set up plot
    fig = plt.figure(figsize=(2.5 * ncols, 2.5 * nrows))
    gs = gridspec.GridSpec(nrows=1, ncols=ncols + 1,
                           width_ratios=[20, ] * ncols + [1, ])
    gs.update(wspace=0.0, hspace=0.0)
    axes = []
    cax = fig.add_subplot(gs[-1])

    for i in range(ncols):
        axes.append(fig.add_subplot(gs[i]))
        axes[-1].loglog()

        if i > 0:
            axes[i].tick_params("y", left=False, right=False, labelleft=False,
                                labelright=False)

    # Loop over redshift bins
    for i, ax in enumerate(axes):

        okinds = np.logical_and(np.logical_and(zs >= zbins[i],
                                               zs < zbins[i + 1]),
                                dens > 0)

        mappable = ax.pcolormesh(birth_density_bins, metal_mass_fraction_bins,
                                 f_th_grid, vmin=0.3, vmax=3)

        H, _, _ = np.histogram2d(den_born[okinds], mets[okinds],
                                 bins=[birth_density_bins,
                                       metal_mass_fraction_bins])

        ax.contour(birth_density_grid, metal_mass_fraction_grid,
                   H.T, levels=6, cmap="magma")

        # Add line showing SF law
        sf_threshold_density = star_formation_parameters["threshold_n0"] * \
            (metal_mass_fraction_bins
             / star_formation_parameters["threshold_Z0"])
        ** (star_formation_parameters["slope"])
        ax.plot(sf_threshold_density, metal_mass_fraction_bins,
                linestyle="dashed", label="SF threshold")
        ax.text(0.1, 0.9, f'${zbins[i]}<{z}<{zbins[i + 1]}$',
                bbox=dict(boxstyle="round,pad=0.3", fc='w', ec="k", lw=1,
                          alpha=0.8),
                transform=ax.transAxes, horizontalalignment='left', fontsize=8)

        for ax in axes:
            ax.set_xlim(10**-3, 10**6.8)
            ax.set_ylim(10**-5.9, 10**0)
            for spine in ax.spines.values():
                spine.set_edgecolor('k')

        axes[i].set_xlabel(r"$Z_{\mathrm{birth}}$")

    # Label y axis
    axes[0].set_ylabel(r"$\rho_{\mathrm{birth}} / $ \mathrm{cm}^{-3}")

    fig.colorbar(im, cax)

    # Save figure
    mkdir("plots/stellar_formprops/")
    fig.savefig("plots/stellar_formprops/stellar_birthden_vs_met_%s.png" % snap,
                bbox_inches="tight")
