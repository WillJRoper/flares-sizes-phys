import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
from flare import plt as flareplt
from utils import mkdir, plot_meidan_stat, age2z
from unyt import mh, cm, Gyr, g, Msun, Mpc

import eagle_IO.eagle_IO as eagle_io


def plot_birth_met(stellar_data, snap, weight_norm, path):

    # Define overdensity bins in log(1+delta)
    ovden_bins = np.arange(-0.3, 0.4, 0.1)

    # Extract arrays
    zs = stellar_data["birth_z"]
    mets = stellar_data["Particle,S_Z_smooth"]
    w = stellar_data["part_weights"]
    part_ovdens = stellar_data["part_ovdens"]

    # Set up the plot
    fig = plt.figure(figsize=(4, 3.5))
    gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[10, 5])
    gs.update(wspace=0.0, hspace=0.0)
    ax = fig.add_subplot(gs[0, 0])
    ax1. fig.add_subplot(gs[1, 0])

    # Remove anomalous values
    okinds = np.ones(mets.size, dtype=bool)

    im = ax.hexbin(zs[okinds], mets[okinds], gridsize=50,
                   mincnt=np.min(w) - (0.1 * np.min(w)),
                   C=w[okinds], norm=LogNorm(),
                   reduce_C_function=np.sum, linewidths=0.2,
                   cmap='viridis')

    # Loop over overdensity bins and plot median curves
    for i in range(ovden_bins[:-1].size):

        # Get boolean indices for this bin
        okinds = np.logical_and(part_ovdens < ovden_bins[i + 1],
                                part_ovdens >= ovden_bins[i])

        plot_meidan_stat(zs[okinds], mets[okinds], w[okinds], ax1,
                         lab=r"$%.1f \leq \log_{10}(1 + \Delta) < %.1f$"
                         % (ovden_bins[i], ovden_bins[i + 1]), color=None)

    ax.set_ylabel(r"$Z_{\mathrm{birth}} + 1$")
    ax.set_xlabel(r"$z_{\mathrm{birth}}$")

    cbar = fig.colorbar(im)
    cbar.set_label("$\sum w_{i}$")

    ax1.legend(loc='upper center',
               bbox_to_anchor=(0.5, -0.2),
               fancybox=True, ncol=2)

    # Save figure
    mkdir("plots/stellar_evo/")
    fig.savefig("plots/stellar_evo/stellar_birthZ_%s.png" % snap,
                bbox_inches="tight")


def plot_birth_den(stellar_data, snap, weight_norm, path):

    # Define overdensity bins in log(1+delta)
    ovden_bins = np.arange(-0.3, 0.4, 0.1)

    # Define redshift
    z = float(snap.split("z")[-1].replace("p", "."))

    # Get the index for this star particle
    s_inds = stellar_data["Particle,S_Index"]

    # Get the region for each galaxy
    regions = stellar_data["regions"]

    # Open region overdensities
    reg_ovdens = np.loadtxt("/cosma7/data/dp004/dc-rope1/FLARES/"
                            "flares/region_overdensity.txt",
                            dtype=float)

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
    part_ovdens = np.zeros(s_inds.size)

    # Extract weights for each particle
    prev_reg = 0
    for igal in range(stellar_data["begin"].size):

        # Extract galaxy range
        b = stellar_data["begin"][igal]
        e = b + stellar_data["Galaxy,S_Length"][igal]
        this_s_inds = s_inds[b: e]

        # Get this galaxies region
        reg = regions[igal]

        # Set this galaxy's region overdensity
        part_ovdens[b: e] = reg_ovdens[reg]

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
                                        noH=True, physicalUnits=True,
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

    # Store the data so we doon't have to recalculate it
    stellar_data["birth_density"] = dens
    stellar_data["birth_z"] = zs
    stellar_data["part_ovdens"] = part_ovdens
    stellar_data["part_weights"] = w

    # Set up the plot
    fig = plt.figure(figsize=(4, 3.5))
    ax = fig.add_subplot(111)

    # Remove anomalous values
    okinds = np.logical_and(zs > 0, dens > 0)

    im = ax.hexbin(zs[okinds], dens[okinds], gridsize=50,
                   mincnt=np.min(w) - (0.1 * np.min(w)),
                   C=w[okinds], norm=LogNorm(),
                   reduce_C_function=np.sum, yscale='log',
                   linewidths=0.2,
                   cmap='viridis')

    # Loop over overdensity bins and plot median curves
    for i in range(ovden_bins[:-1].size):

        # Get boolean indices for this bin
        okinds = np.logical_and(part_ovdens < ovden_bins[i + 1],
                                part_ovdens >= ovden_bins[i])
        okinds = np.logical_and(np.logical_and(zs > 0, dens > 0),
                                okinds)

        plot_meidan_stat(zs[okinds], dens[okinds], w[okinds], ax,
                         lab=r"$%.1f \leq \log_{10}(1 + \Delta) < %.1f$"
                         % (ovden_bins[i], ovden_bins[i + 1]), color=None)

    ax.set_ylabel(r"$n_{\mathrm{H}} / \mathrm{cm}^{-3}$")
    ax.set_xlabel(r"$z_{\mathrm{birth}}$")

    cbar = fig.colorbar(im)
    cbar.set_label("$\sum w_{i}$")

    ax.legend(loc='upper center',
              bbox_to_anchor=(0.5, -0.2),
              fancybox=True, ncol=2)

    # Save figure
    mkdir("plots/stellar_evo/")
    fig.savefig("plots/stellar_evo/stellar_birthden_%s.png" % snap,
                bbox_inches="tight")

    return stellar_data


def plot_birth_den_vs_met(stellar_data, snap, weight_norm, path):

    # Define redshift bins
    zbins = list(np.arange(5, 12.5, 2.5))
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
    birth_density_bins = np.logspace(-2.9, 6.8, number_of_bins)
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

    # Extract arrays
    zs = stellar_data["birth_z"]
    dens = stellar_data["birth_density"]
    mets = stellar_data["Particle,S_Z_smooth"]
    w = stellar_data["part_weights"]

    # Set up the plot
    ncols = len(zbins) - 1

    # Set up plot
    fig = plt.figure(figsize=(2.5 * ncols, 2.5))
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

        H, _, _ = np.histogram2d(dens[okinds], mets[okinds],
                                 bins=[birth_density_bins,
                                       metal_mass_fraction_bins])

        ax.contour(birth_density_grid, metal_mass_fraction_grid,
                   H.T, levels=6, cmap="magma")

        # Add line showing SF law
        sf_threshold_density = star_formation_parameters["threshold_n0"] * \
            (metal_mass_fraction_bins
             / star_formation_parameters["threshold_Z0"]) \
            ** (star_formation_parameters["slope"])
        ax.plot(sf_threshold_density, metal_mass_fraction_bins,
                linestyle="dashed", label="SF threshold")
        ax.text(0.95, 0.9, f'${zbins[i]}<z<{zbins[i + 1]}$',
                bbox=dict(boxstyle="round,pad=0.3", fc='w', ec="k", lw=1,
                          alpha=0.8),
                transform=ax.transAxes, horizontalalignment='right', fontsize=8)

        for ax in axes:
            ax.set_xlim(10**-2.9, 10**6.8)
            ax.set_ylim(10**-5.9, 10**0)
            for spine in ax.spines.values():
                spine.set_edgecolor('k')

        axes[i].set_xlabel(r"$n_{\mathrm{H}} / \mathrm{cm}^{-3}$")

    # Label y axis
    axes[0].set_ylabel(r"$Z_{\mathrm{birth}}$")

    cbar = fig.colorbar(mappable, cax)
    cbar.set_label(r"$f_\mathrm{th}$")

    # Save figure
    mkdir("plots/stellar_formprops/")
    fig.savefig("plots/stellar_formprops/stellar_birthden_vs_met_%s.png" % snap,
                bbox_inches="tight")


def plot_gal_birth_den_vs_met(stellar_data, snap, weight_norm, path):

    # Define redshift bins
    zbins = list(np.arange(5, 12.5, 2.5))
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
    birth_density_bins = np.logspace(-2.9, 6.8, number_of_bins)
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

    # Extract arrays
    dens = stellar_data["birth_density"]
    mets = stellar_data["Particle,S_Z_smooth"]
    w = stellar_data["part_weights"]
    radii = stellar_data["radii"]

    # Define how many galaxies we have
    ngal = stellar_data["begin"].size

    # Set up arrays to store results
    gal_bdens = np.zeros(ngal)
    gal_bmet = np.zeros(ngal)
    gal_w = np.zeros(ngal)

    # Loop over each galaxy calculating initial mass weighted properties
    for igal in range(ngal):

        # Extract galaxy range
        b = stellar_data["begin"][igal]
        e = b + stellar_data["Galaxy,S_Length"][igal]
        ini_mass = stellar_data["Particle,S_MassInitial"][b: e]
        hmr = stellar_data["HMRs"][igal]
        rs = radii[b: e]

        # Calculate initial mass weighted properties
        okinds = np.logical_and(rs != 0, rs <= hmr)
        if np.sum(ini_mass[okinds]) == 0:
            continue
        gal_bdens[igal] = np.average(
            dens[b: e][okinds], weights=ini_mass[okinds])
        gal_bmet[igal] = np.average(
            mets[b: e][okinds], weights=ini_mass[okinds])
        gal_w[igal] = w[b: e][0]

    # Export this data
    stellar_data["gal_birth_density"] = gal_bdens

    # Set up the plot
    ncols = len(zbins) - 1

    # Set up the plot
    fig = plt.figure(figsize=(3.5, 3.5))
    ax = fig.add_subplot(111)

    # Remove anomalous values
    okinds = np.logical_and(gal_bdens > 0, gal_bmet + 1 > 0)

    im = ax.hexbin(gal_bdens[okinds], gal_bmet[okinds] + 1, gridsize=50,
                   mincnt=np.min(w) - (0.1 * np.min(w)),
                   C=gal_w[okinds], norm=weight_norm,
                   reduce_C_function=np.sum, yscale='log', xscale="log",
                   linewidths=0.2,
                   cmap='viridis')

    ax.set_xlabel(r"$\bar{n}_{\mathrm{H}} / \mathrm{cm}^{-3}$")
    ax.set_ylabel(r"$\bar{Z}_{\mathrm{birth}} + 1$")

    cbar = fig.colorbar(im)
    cbar.set_label("$\sum w_{i}$")

    # Save figure
    mkdir("plots/stellar_formprops/")
    fig.savefig("plots/stellar_formprops/galaxy_stellar_birthden_%s.png" % snap,
                bbox_inches="tight")

    return stellar_data
