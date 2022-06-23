import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
import matplotlib.gridspec as gridspec
from brokenaxes import brokenaxes
from flare import plt as flareplt
from utils import mkdir, plot_meidan_stat, age2z, get_nonmaster_evo_data
from unyt import mh, cm, Gyr, g, Msun, Mpc
from astropy.cosmology import Planck18 as cosmo, z_at_value
import astropy.units as u
import astropy.constants as const
import pandas as pd

import eagle_IO.eagle_IO as eagle_io


def plot_birth_met(stellar_data, snap, weight_norm, path):

    # Define intial FLARES path
    ini_path = "/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/G-EAGLE_<reg>/data/"

    # Open region overdensities
    reg_ovdens = np.loadtxt("/cosma7/data/dp004/dc-rope1/FLARES/"
                            "flares/region_overdensity.txt",
                            dtype=float)
    print(reg_ovdens)

    # Load weights
    df = pd.read_csv('../weight_files/weights_grid.txt')
    weights = np.array(df['weights'])

    print(weights)

    # Define regions
    regions = []
    for reg in range(0, 40):
        if reg < 10:
            regions.append('0' + str(reg))
        else:
            regions.append(str(reg))

    # Define overdensity bins in log(1+delta)
    ovden_bins = np.arange(-0.3, 0.4, 0.1)
    eagle_z_bins = np.arange(0.0, 16.0, 1.0)
    flares_z_bins = np.arange(4.5, 15.5, 1.0)

    # Define lists to store data
    zs = []
    mets = []
    ovdens = []
    w = []
    for reg in regions:
        path = ini_path.replace("<reg>", reg)

        reg_zs, reg_mets = get_nonmaster_evo_data(
            path, snap, y_key="PartType4/SmoothedMetallicity")
        zs.extend(reg_zs)
        mets.extend(reg_mets)
        ovdens.extend(np.full(reg_zs.size, reg_ovdens[int(reg)], dtype=float))
        w.extend(np.full(reg_zs.size, weights[int(reg)], dtype=float))

    # Convert to arrays
    zs = np.array(zs)
    mets = np.array(mets)
    part_ovdens = np.array(ovdens)
    w = np.array(w)

    # Get eagle data
    agndt9_path = '/cosma7/data/Eagle/ScienceRuns/Planck1/L0050N0752/PE/AGNdT9/data/'
    agndt9_zs, agndt9_mets = get_nonmaster_evo_data(
        agndt9_path, "028_z000p000", y_key="PartType4/SmoothedMetallicity")

    # Get eagle data
    ref_path = "/cosma7/data/Eagle/ScienceRuns/Planck1/L0100N1504/PE/REFERENCE/data"
    eagle_zs, eagle_mets = get_nonmaster_evo_data(
        ref_path, "028_z000p000", y_key="PartType4/SmoothedMetallicity")

    # Set up the plotd
    fig = plt.figure(figsize=(3.5, 3.5))
    ax = fig.add_subplot(111)

    plot_meidan_stat(eagle_zs, eagle_mets, np.ones(eagle_mets.size), ax,
                     lab=r"EAGLE-REF", bins=eagle_z_bins, color=None,
                     ls="dotted")
    plot_meidan_stat(agndt9_zs, agndt9_mets, np.ones(agndt9_mets.size), ax,
                     lab=r"EAGLE-AGNdT9", bins=eagle_z_bins, color=None,
                     ls="--")

    # Loop over overdensity bins and plot median curves
    for i in range(ovden_bins[:-1].size):

        # Get boolean indices for this bin
        okinds = np.logical_and(part_ovdens < ovden_bins[i + 1],
                                part_ovdens >= ovden_bins[i])
        okinds = np.logical_and(np.logical_and(zs > 0, mets >= 0),
                                okinds)

        plot_meidan_stat(zs[okinds], mets[okinds],
                         w[okinds], ax,
                         lab=r"$%.1f \leq \log_{10}(1 + \Delta) < %.1f$"
                         % (ovden_bins[i], ovden_bins[i + 1]),
                         bins=flares_z_bins,
                         color=None)

    ax.set_ylim(0.0, 0.005)

    ax.set_ylabel(r"$Z_{\mathrm{birth}}$")
    ax.set_xlabel(r"$z_{\mathrm{birth}}$")

    ax.legend(loc='upper center',
              bbox_to_anchor=(0.5, -0.2),
              fancybox=True, ncol=2)

    # Save figure
    mkdir("plots/stellar_evo/")
    fig.savefig("plots/stellar_evo/stellar_birthZ_%s.png" % snap,
                bbox_inches="tight")


def plot_birth_den(stellar_data, snap, weight_norm, path):

    # Define intial FLARES path
    ini_path = "/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/G-EAGLE_<reg>/data/"

    # Open region overdensities
    reg_ovdens = np.loadtxt("/cosma7/data/dp004/dc-rope1/FLARES/"
                            "flares/region_overdensity.txt",
                            dtype=float)
    print(reg_ovdens)

    # Load weights
    df = pd.read_csv('../weight_files/weights_grid.txt')
    weights = np.array(df['weights'])

    print(weights)

    # Define regions
    regions = []
    for reg in range(0, 40):
        if reg < 10:
            regions.append('0' + str(reg))
        else:
            regions.append(str(reg))

    # Define overdensity bins in log(1+delta)
    ovden_bins = np.arange(-0.3, 0.4, 0.1)
    eagle_z_bins = np.arange(0.0, 16.0, 1.0)
    flares_z_bins = np.arange(4.5, 15.5, 1.0)

    # Define lists to store data
    zs = []
    dens = []
    ovdens = []
    w = []
    for reg in regions:
        path = ini_path.replace("<reg>", reg)

        reg_zs, reg_dens = get_nonmaster_evo_data(
            path, snap, y_key="PartType4/BirthDensity")
        zs.extend(reg_zs)
        dens.extend(reg_dens)
        ovdens.extend(np.full(reg_zs.size, reg_ovdens[int(reg)], dtype=float))
        w.extend(np.full(reg_zs.size, weights[int(reg)], dtype=float))

    # Convert to arrays
    zs = np.array(zs)
    dens = np.array(dens)
    part_ovdens = np.array(ovdens)
    w = np.array(w)

    # Get eagle data
    agndt9_path = '/cosma7/data/Eagle/ScienceRuns/Planck1/L0050N0752/PE/AGNdT9/data/'
    agndt9_zs, agndt9_dens = get_nonmaster_evo_data(
        agndt9_path, "028_z000p000", y_key="PartType4/BirthDensity")

    # Get eagle data
    ref_path = "/cosma7/data/Eagle/ScienceRuns/Planck1/L0100N1504/PE/REFERENCE/data"
    eagle_zs, eagle_dens = get_nonmaster_evo_data(
        ref_path, "028_z000p000", y_key="PartType4/BirthDensity")

    # Set up the plot
    fig = plt.figure(figsize=(3.5, 3.5))
    ax = fig.add_subplot(111)
    ax.semilogy()

    plot_meidan_stat(eagle_zs, eagle_dens,
                     np.ones(eagle_dens.size), ax,
                     lab=r"EAGLE-REF", bins=eagle_z_bins, color=None,
                     ls="dotted")
    plot_meidan_stat(agndt9_zs, agndt9_dens,
                     np.ones(agndt9_dens.size), ax,
                     lab=r"EAGLE-AGNdT9", bins=eagle_z_bins, color=None,
                     ls="--")

    # Loop over overdensity bins and plot median curves
    for i in range(ovden_bins[:-1].size):

        # Get boolean indices for this bin
        okinds = np.logical_and(part_ovdens < ovden_bins[i + 1],
                                part_ovdens >= ovden_bins[i])
        okinds = np.logical_and(np.logical_and(zs > 0, dens > 0),
                                okinds)

        plot_meidan_stat(zs[okinds], dens[okinds], w[okinds], ax,
                         lab=r"$%.1f \leq \log_{10}(1 + \Delta) < %.1f$"
                         % (ovden_bins[i], ovden_bins[i + 1]),
                         bins=flares_z_bins, color=None)

    ax.set_ylabel(r"$n_{\mathrm{H}} / \mathrm{cm}^{-3}$")
    ax.set_xlabel(r"$z_{\mathrm{birth}}$")

    # cbar = fig.colorbar(im)
    # cbar.set_label("$\sum w_{i}$")

    ax.legend(loc='upper center',
              bbox_to_anchor=(0.5, -0.2),
              fancybox=True, ncol=2)

    # Save figure
    mkdir("plots/stellar_evo/")
    fig.savefig("plots/stellar_evo/stellar_birthden_%s.png" % snap,
                bbox_inches="tight")

    return stellar_data


def plot_sfr_evo(stellar_data, snap):

    # Define overdensity bins in log(1+delta)
    ovden_bins = np.arange(-0.3, 0.4, 0.1)
    eagle_z_bins = np.arange(0.5, 20.5, 0.5)
    flares_z_bins = np.arange(4.5, 23.5, 0.5)

    # Extract arrays
    zs = stellar_data["birth_z"]
    ms = stellar_data["Particle,S_MassInitial"] * 10 ** 10
    part_ovdens = stellar_data["part_ovdens"]
    part_nstar = stellar_data["part_nstar"]

    # Remove particles with 0 weight (these are not in a galaxy
    # we are including)
    okinds = part_nstar >= 100
    ms = ms[okinds]
    zs = zs[okinds]
    part_ovdens = part_ovdens[okinds]
    part_nstar = part_nstar[okinds]

    # Get eagle data
    ref_path = '/cosma7/data/Eagle/ScienceRuns/Planck1/L0050N0752/PE/AGNdT9/data/'
    eagle_aborn = eagle_io.read_array('PARTDATA', ref_path, '027_z000p101',
                                      'PartType4/StellarFormationTime',
                                      noH=True,
                                      physicalUnits=True,
                                      numThreads=8)
    pre_eagle_ms = eagle_io.read_array('PARTDATA', ref_path, '027_z000p101',
                                       'PartType4/InitialMass',
                                       noH=True,
                                       physicalUnits=True,
                                       numThreads=8) * 10 ** 10
    pre_eagle_zs = 1 / eagle_aborn - 1
    subgrps = eagle_io.read_array('SUBFIND', ref_path, '028_z000p000',
                                  'Subhalo/SubGroupNumber', noH=True,
                                  physicalUnits=True,
                                  numThreads=8)
    grps = eagle_io.read_array('SUBFIND', ref_path, '028_z000p000',
                               'Subhalo/GroupNumber', noH=True,
                               physicalUnits=True,
                               numThreads=8)
    nstars = eagle_io.read_array('SUBFIND', ref_path, '028_z000p000',
                                 'Subhalo/SubLengthType', noH=True,
                                 physicalUnits=True,
                                 numThreads=8)[:, 4]
    part_subgrp = eagle_io.read_array('PARTDATA', ref_path, '028_z000p000',
                                      'PartType4/SubGroupNumber',
                                      noH=True,
                                      physicalUnits=True,
                                      numThreads=8)
    part_grp = eagle_io.read_array('PARTDATA', ref_path, '028_z000p000',
                                   'PartType4/GroupNumber', noH=True,
                                   physicalUnits=True,
                                   numThreads=8)

    # Clean up eagle data to remove galaxies with nstar < 100
    nstar_dict = {}
    for (ind, grp), subgrp in zip(enumerate(grps), subgrps):

        if grp == 2**30 or subgrp == 2**30:
            continue

        # Skip particles not in a galaxy
        if nstars[ind] >= 100:
            nstar_dict[(grp, subgrp)] = nstars[ind]

    # Now get the stars we want
    eagle_zs = []
    eagle_ms = []
    for ind in range(eagle_aborn.size):

        # Get grp and subgrp
        grp, subgrp = part_grp[ind], part_subgrp[ind]

        if (grp, subgrp) in nstar_dict:
            eagle_zs.append(pre_eagle_zs[ind])
            eagle_ms.append(pre_eagle_ms[ind])

    # Convert to arrays
    eagle_zs = np.array(eagle_zs)
    eagle_ms = np.array(eagle_ms)

    # Set up the plotd
    fig = plt.figure(figsize=(3.5, 3.5))
    ax = fig.add_subplot(111)
    ax.semilogy()

    # Loop over overdensity bins and plot median curves
    for i in range(ovden_bins[:-1].size):

        # Get boolean indices for this bin
        okinds = np.logical_and(part_ovdens < ovden_bins[i + 1],
                                part_ovdens >= ovden_bins[i])
        okinds = np.logical_and(np.logical_and(zs > 0, ms >= 0),
                                okinds)

        # Get SFRs
        sfrs = []
        plt_zs = []
        for z_low in flares_z_bins[:-1]:

            z_high = z_at_value(cosmo.age, cosmo.age(z_low) - (100 * u.Myr),
                                zmin=0, zmax=50)

            zokinds = np.logical_and(np.logical_and(zs < z_high, zs >= z_low),
                                     okinds)

            sfrs.append(np.sum(ms[zokinds]) / 100)  # M_sun / Myr
            plt_zs.append(z_low)

        ax.plot(plt_zs, sfrs,
                label=r"$%.1f \leq \log_{10}(1 + \Delta) < %.1f$"
                % (ovden_bins[i], ovden_bins[i + 1]))

    # Get SFRs
    sfrs = []
    plt_zs = []
    for z_low in eagle_z_bins[:-1]:

        z_high = z_at_value(cosmo.age, cosmo.age(z_low) - (100 * u.Myr),
                            zmin=0, zmax=50)

        zokinds = np.logical_and(eagle_zs < z_high, eagle_zs >= z_low)

        sfrs.append(np.sum(eagle_ms[zokinds]) / 100)  # M_sun / Myr
        plt_zs.append(z_low)

    ax.plot(plt_zs, sfrs, label=r"EAGLE-AGNdT9", ls="--")

    ax.set_ylabel(
        r"$\mathrm{SFR}_{100} / [\mathrm{M}_\odot\mathrm{Myr}^{-1}]$")
    ax.set_xlabel(r"$z_{\mathrm{birth}}$")

    ax.legend(loc='upper center',
              bbox_to_anchor=(0.5, -0.2),
              fancybox=True, ncol=2)

    # Save figure
    mkdir("plots/stellar_evo/")
    fig.savefig("plots/stellar_evo/stellar_SFRevo_%s.png" % snap,
                bbox_inches="tight")


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
    fig = plt.figure(figsize=(3.5 * ncols, 3.5))
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
                                 f_th_grid, vmin=0.3, vmax=3, cmap="inferno")

        H, _, _ = np.histogram2d(dens[okinds], mets[okinds],
                                 bins=[birth_density_bins,
                                       metal_mass_fraction_bins])

        ax.contour(birth_density_grid, metal_mass_fraction_grid,
                   H.T, levels=6, cmap="viridis")

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


def plot_subgrid_birth_den_vs_met():

    # Lets define our colormap
    colors1 = mpl.cm.get_cmap('inferno')(np.linspace(0, 1., 128))
    colors2 = mpl.cm.get_cmap('viridis_r')(np.linspace(0, 1., 128))
    colors_combined = []
    colors_combined.extend(colors1)
    colors_combined.extend(colors2)
    colors = list(zip(np.linspace(0, 1.0, 256), colors_combined))
    cmap = mpl.colors.LinearSegmentedColormap.from_list('mycmap', colors)

    # Set up some variables we need
    fmaxs = [3, 4, 6, 10]
    ncols = len(fmaxs)

    # Set up plot
    fig = plt.figure(figsize=(3.5 * ncols, 3.5))
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

    for i, fmax in enumerate(fmaxs):

        # Define EAGLE subgrid  parameters
        parameters = {"f_th,min": 0.3,
                      "f_th,max": fmax,
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

        mappable = axes[i].pcolormesh(birth_density_bins,
                                      metal_mass_fraction_bins,
                                      f_th_grid,
                                      norm=TwoSlopeNorm(vmin=0.3,
                                                        vcenter=3,
                                                        vmax=10),
                                      cmap=cmap
                                      )

        for slope in [-0.64, 0]:

            star_formation_parameters = {"threshold_Z0": 0.002,
                                         "threshold_n0": 0.1,
                                         "slope": slope}

            # Add line showing SF law
            sf_threshold_density = star_formation_parameters["threshold_n0"] * \
                (metal_mass_fraction_bins
                 / star_formation_parameters["threshold_Z0"]) \
                ** (star_formation_parameters["slope"])
            axes[i].plot(sf_threshold_density, metal_mass_fraction_bins,
                         linestyle="dashed", label="SF threshold: slope=%.2f" % slope)

        axes[i].set_xlabel(r"$n_{\mathrm{H}} / \mathrm{cm}^{-3}$")

        for ax in axes:
            ax.set_xlim(10**-2.9, 10**6.8)
            ax.set_ylim(10**-5.9, 10**0)
            for spine in ax.spines.values():
                spine.set_edgecolor('k')

        # Label each panel
        axes[i].text(0.95, 0.9, r'$f_{\mathrm{th,max}}=%.1f$' % fmax,
                     bbox=dict(boxstyle="round,pad=0.3", fc='w', ec="k", lw=1,
                               alpha=0.8),
                     transform=axes[i].transAxes, horizontalalignment='right', fontsize=8)

    # Label y axis
    axes[0].set_ylabel(r"$Z_{\mathrm{birth}}$")

    cbar = fig.colorbar(mappable, cax, ticks=[0.3, 3, 4, 6, 10])
    cbar.set_label(r"$f_\mathrm{th}$")

    axes[2].legend(loc='upper center',
                   bbox_to_anchor=(0.0, -0.2),
                   fancybox=True, ncol=2)

    # Save figure
    mkdir("plots/stellar_formprops/")
    fig.savefig("plots/stellar_formprops/stellar_birthden_vs_met_subgrid.png",
                bbox_inches="tight")


def plot_eagle_birth_den_vs_met(stellar_data, snap, weight_norm, path):

    # Define redshift bins
    zbins = list(np.arange(0, 12.5, 2.5))
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
    ref_path = '/cosma7/data/Eagle/ScienceRuns/Planck1/L0050N0752/PE/AGNdT9/data/'
    eagle_aborn = eagle_io.read_array('PARTDATA', ref_path, '028_z000p000',
                                      'PartType4/StellarFormationTime',
                                      numThreads=8)
    pre_eagle_zs = 1 / eagle_aborn - 1
    pre_eagle_mets = eagle_io.read_array('PARTDATA', ref_path, '028_z000p000',
                                         'PartType4/SmoothedMetallicity',
                                         numThreads=8)
    pre_eagle_dens = (eagle_io.read_array('PARTDATA', ref_path, '028_z000p000',
                                          "PartType4/BirthDensity",
                                          numThreads=8, noH=True,
                                          physicalUnits=True) * 10**10
                      * Msun / Mpc ** 3 / mh).to(1 / cm ** 3).value
    subgrps = eagle_io.read_array('SUBFIND', ref_path, '028_z000p000',
                                  'Subhalo/SubGroupNumber', noH=True,
                                  physicalUnits=True,
                                  numThreads=8)
    grps = eagle_io.read_array('SUBFIND', ref_path, '028_z000p000',
                               'Subhalo/GroupNumber', noH=True,
                               physicalUnits=True,
                               numThreads=8)
    nstars = eagle_io.read_array('SUBFIND', ref_path, '028_z000p000',
                                 'Subhalo/SubLengthType', noH=True,
                                 physicalUnits=True,
                                 numThreads=8)[:, 4]
    part_subgrp = eagle_io.read_array('PARTDATA', ref_path, '028_z000p000',
                                      'PartType4/SubGroupNumber',
                                      noH=True,
                                      physicalUnits=True,
                                      numThreads=8)
    part_grp = eagle_io.read_array('PARTDATA', ref_path, '028_z000p000',
                                   'PartType4/GroupNumber', noH=True,
                                   physicalUnits=True,
                                   numThreads=8)

    # Clean up eagle data to remove galaxies with nstar < 100
    nstar_dict = {}
    for (ind, grp), subgrp in zip(enumerate(grps), subgrps):

        # Skip particles not in a galaxy
        if nstars[ind] >= 100:
            nstar_dict[(grp, subgrp)] = nstars[ind]

    # Now get the stars we want
    eagle_zs = []
    eagle_dens = []
    eagle_mets = []
    for ind in range(eagle_aborn.size):

        # Get grp and subgrp
        grp, subgrp = part_grp[ind], part_subgrp[ind]

        if (grp, subgrp) in nstar_dict:
            eagle_zs.append(pre_eagle_zs[ind])
            eagle_dens.append(pre_eagle_dens[ind])
            eagle_mets.append(pre_eagle_mets[ind])

    # Convert to arrays
    eagle_zs = np.array(eagle_zs)
    eagle_dens = np.array(eagle_dens)
    eagle_mets = np.array(eagle_mets)

    # Get fake weights
    eagle_w = np.ones(eagle_dens.size)

    zs = np.concatenate((stellar_data["birth_z"], eagle_zs))
    dens = np.concatenate((stellar_data["birth_density"], eagle_dens))
    mets = np.concatenate((stellar_data["Particle,S_Z_smooth"], eagle_mets))
    w = np.concatenate((stellar_data["part_weights"], eagle_w))

    # Set up the plot
    ncols = len(zbins) - 1

    # Set up plot
    fig = plt.figure(figsize=(3.5 * ncols, 3.5))
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
                                 f_th_grid, vmin=0.3, vmax=3, cmap="inferno")

        H, _, _ = np.histogram2d(dens[okinds], mets[okinds],
                                 bins=[birth_density_bins,
                                       metal_mass_fraction_bins])

        ax.contour(birth_density_grid, metal_mass_fraction_grid,
                   H.T, levels=6, cmap="viridis")

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

    # Set up arrays to store results
    gal_bdens = stellar_data["gal_birth_density"]
    gal_bmet = stellar_data["gal_birth_density"]
    gal_w = stellar_data["gal_weight"]

    # Set up the plot
    fig = plt.figure(figsize=(3.5, 3.5))
    ax = fig.add_subplot(111)

    # Remove anomalous values
    okinds = np.logical_and(gal_bdens > 0, gal_bmet + 1 > 0)

    im = ax.hexbin(gal_bdens[okinds], gal_bmet[okinds] + 1, gridsize=50,
                   mincnt=np.min(gal_w) - (0.1 * np.min(gal_w)),
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


def virial_temp(m, mu, z):
    T = 4 * 10 ** 4 * (mu / 1.2) * (m / (10 ** 8 / 0.6777)) ** (1 + z / 10)
    return T


def plot_virial_temp():

    # Define arrays of masses and sizes
    ms = np.logspace(7, 13, 1000)
    zs = np.linspace(5, 20, 256)

    # Set up colormap
    cmap = mpl.cm.plasma
    norm = LogNorm(vmin=np.min(zs), vmax=np.max(zs))

    # Set up the plot
    fig = plt.figure(figsize=(3.5, 3.5))
    gs = gridspec.GridSpec(nrows=2, ncols=2,
                           width_ratios=[20, 1])
    gs.update(wspace=0.0, hspace=0.0)
    ax = fig.add_subplot(gs[:, 0])
    cax = fig.add_subplot(gs[:, -1])
    ax.semilogy()

    # Loop over hmrs calculating virial temperatures
    for z in zs:
        ts = virial_temp(ms, hmr * 2, mu=1.2, z=z)
        ax.plot(ms, ts, color=cmap(norm(z)))

    # Set labels
    ax.set_xlabel("$M_\mathrm{tot} / M_\odot$")
    ax.set_ylabel("$T_{\mathrm{vir}} /$ [K]")

    # Make colorbar
    cb1 = mpl.colorbar.ColorbarBase(cax, cmap=cmap,
                                    norm=norm)
    cb1.set_label("$R_{1/2} /$ [pkpc]")

    # Save figure
    mkdir("plots/stellar_formprops/")
    fig.savefig("plots/stellar_formprops/virial_temp.png",
                bbox_inches="tight")
