from flare import plt as flareplt
import os

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from utils import calc_3drad, calc_light_mass_rad, mkdir

os.environ['FLARE'] = '/cosma7/data/dp004/dc-wilk2/flare'
mpl.use('Agg')


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


def plot_stellar_hmr(stellar_data, snap, weight_norm):

    # Define arrays to store computations
    hmrs = np.zeros(len(stellar_data["begin"]))
    mass = np.zeros(len(stellar_data["begin"]))
    den_hmr = np.zeros(len(stellar_data["begin"]))
    w = np.zeros(len(stellar_data["begin"]))
    radii = np.zeros(len(stellar_data["Particle,S_Mass"]))

    # Loop over galaxies and calculate stellar HMR
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
        radii[b: b + l][app] = rs

        # Compute HMR
        hmr = calc_light_mass_rad(rs, ms, radii_frac=0.5)

        # Compute stellar density within HMR
        den_hmr[igal] = (np.sum(ms[rs <= hmr]) * 10 ** 10
                         / (4 / 3 * np.pi * hmr ** 3))

        # Store results
        mass[igal] = np.sum(ms) * 10 ** 10
        hmrs[igal] = hmr
        w[igal] = stellar_data["weights"][igal]

    # Store half mass radii and stellar density within HMR
    stellar_data["HMRs"] = hmrs
    stellar_data["HMR_Density"] = den_hmr
    stellar_data["density_lim"] = 10 ** 9
    stellar_data["radii"] = radii
    stellar_data["mass"] = mass
    stellar_data["weight"] = w

    # Remove galaxies without stars
    okinds = np.logical_and(mass > 0, hmrs > 0)
    print("Galaxies before spurious cut: %d" % mass.size)
    mass = mass[okinds]
    hmrs = hmrs[okinds]
    den_hmr = den_hmr[okinds]
    w = w[okinds]
    print("Galaxies after spurious cut: %d" % mass.size)

    # Set up plot
    fig = plt.figure(figsize=(3.5, 3.5))
    ax = fig.add_subplot(111)
    ax.loglog()

    # Define boolean indices for each population
    com_pop = den_hmr >= stellar_data["density_lim"]
    diff_pop = ~com_pop

    # Plot stellar_data
    im = ax.hexbin(mass[com_pop], hmrs[com_pop], gridsize=50,
                   mincnt=np.min(w) - (0.1 * np.min(w)),
                   C=w[com_pop],
                   reduce_C_function=np.sum, xscale='log', yscale='log',
                   norm=weight_norm, linewidths=0.2, cmap='viridis')
    ax.hexbin(mass[diff_pop], hmrs[diff_pop], gridsize=50,
              mincnt=np.min(w) - (0.1 * np.min(w)),
              C=w[diff_pop],
              reduce_C_function=np.sum, xscale='log', yscale='log',
              norm=weight_norm, linewidths=0.2, cmap='Greys')

    # Label axes
    ax.set_xlabel("$M_\star / M_\odot$")
    ax.set_ylabel("$R_{1/2} / [\mathrm{pkpc}]$")

    cbar = fig.colorbar(im)
    cbar.set_label("$\sum w_{i}$")

    # Save figure
    mkdir("plots/stellar_hmr/")
    fig.savefig("plots/stellar_hmr/stellar_hmr_%s.png" % snap,
                bbox_inches="tight")

    plt.close(fig)

    return stellar_data


def plot_stellar_hmr_withcut(stellar_data, snap, weight_norm):

    # Define arrays to store computations
    hmrs = stellar_data["HMRs"]
    mass = stellar_data["mass"]
    den_hmr = stellar_data["gal_birth_density"]
    w = stellar_data["weight"]

    # Remove galaxies without stars
    okinds = np.logical_and(mass > 0, hmrs > 0)
    print("Galaxies before spurious cut: %d" % mass.size)
    mass = mass[okinds]
    hmrs = hmrs[okinds]
    den_hmr = den_hmr[okinds]
    w = w[okinds]
    print("Galaxies after spurious cut: %d" % mass.size)

    # Set up plot
    fig = plt.figure(figsize=(3.5, 3.5))
    ax = fig.add_subplot(111)
    ax.loglog()

    # Define boolean indices for each population
    com_pop = den_hmr >= 10**2.5
    diff_pop = ~com_pop

    # Plot stellar_data
    im = ax.hexbin(mass[com_pop], hmrs[com_pop], gridsize=50,
                   mincnt=np.min(w) - (0.1 * np.min(w)),
                   C=w[com_pop],
                   reduce_C_function=np.sum, xscale='log', yscale='log',
                   norm=weight_norm, linewidths=0.2, cmap='viridis')
    ax.hexbin(mass[diff_pop], hmrs[diff_pop], gridsize=50,
              mincnt=np.min(w) - (0.1 * np.min(w)),
              C=w[diff_pop],
              reduce_C_function=np.sum, xscale='log', yscale='log',
              norm=weight_norm, linewidths=0.2, cmap='Greys')

    # Label axes
    ax.set_xlabel("$M_\star / M_\odot$")
    ax.set_ylabel("$R_{1/2} / [\mathrm{pkpc}]$")

    cbar = fig.colorbar(im)
    cbar.set_label("$\sum w_{i}$")

    # Save figure
    mkdir("plots/stellar_hmr/")
    fig.savefig("plots/stellar_hmr/stellar_hmr_%s.png" % snap,
                bbox_inches="tight")

    plt.close(fig)


def plot_stellar_gas_hmr_comp(stellar_data, gas_data, snap, weight_norm):

    # Define arrays to store computations
    s_hmrs = stellar_data["HMRs"]
    s_den_hmr = stellar_data["HMR_Density"]
    g_hmrs = np.zeros(len(gas_data["begin"]))
    w = np.zeros(len(stellar_data["begin"]))
    den_hmr = stellar_data["gal_birth_density"]
    m_age = np.zeros(len(stellar_data["begin"]))

    # Loop over galaxies and extract their weighting
    for (igal, b), l in zip(enumerate(stellar_data["begin"]),
                            stellar_data["Galaxy,S_Length"]):

        if l < 100:
            continue

        w[igal] = stellar_data["weights"][igal]

    # Loop over galaxies and calculate gas HMR
    for (igal, b), l in zip(enumerate(gas_data["begin"]),
                            gas_data["Galaxy,G_Length"]):

        if l < 100:
            continue

        # Get this galaxy's gas_data
        app = gas_data["Particle/Apertures/Gas,30"][b: b + l]
        cop = gas_data["Galaxy,COP"][igal]
        ms = gas_data["Particle,G_Mass"][b: b + l][app]
        pos = gas_data["Particle,G_Coordinates"][b: b + l, :][app]
        age = stellar_data["Particle,S_Age"][b: b + l][app]
        ini_ms = stellar_data["Particle,S_MassInitial"][b: b + l][app]

        # Compute particle radii
        rs = calc_3drad(pos - cop)

        # Compute HMR
        hmr = calc_light_mass_rad(rs, ms, radii_frac=0.5)

        # Store results
        g_hmrs[igal] = hmr
        w[igal] = gas_data["weights"][igal]
        if np.sum(ini_ms) > 0:
            m_age[igal] = np.average(age, weights=ini_ms)

    # Store gas HMRs and gas density within HMR
    gas_data["HMRs"] = g_hmrs

    # Remove galaxies without stars
    okinds = np.logical_and(g_hmrs > 0, s_hmrs > 0)
    print("Galaxies before spurious cut: %d" % s_hmrs.size)
    s_hmrs = s_hmrs[okinds]
    g_hmrs = g_hmrs[okinds]
    s_den_hmr = s_den_hmr[okinds]
    m_age = m_age[okinds]
    w = w[okinds]
    print("Galaxies after spurious cut: %d" % s_hmrs.size)

    # Set up plot
    ncols = 2
    fig = plt.figure(figsize=(2.5 * ncols, 2.5))
    gs = gridspec.GridSpec(nrows=1, ncols=ncols + 1,
                           width_ratios=[20, ] * ncols + [1, ])
    gs.update(wspace=0.0, hspace=0.0)
    ax = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    cax = fig.add_subplot(gs[-1])

    # Define boolean indices for each population
    com_pop = s_den_hmr >= 10**2.5
    diff_pop = ~com_pop

    # Plot stellar_data
    im = ax.hexbin(s_hmrs[com_pop], g_hmrs[com_pop], gridsize=50,
                   mincnt=np.min(w) - (0.1 * np.min(w)),
                   C=w[com_pop], extent=[-1, 1.3, -1, 1.3],
                   reduce_C_function=np.sum, xscale='log', yscale='log',
                   norm=weight_norm, linewidths=0.2, cmap='viridis')
    # Plot stellar_data
    im1 = ax.hexbin(s_hmrs[diff_pop], g_hmrs[diff_pop], gridsize=50,
                    mincnt=np.min(w) - (0.1 * np.min(w)),
                    C=w[diff_pop], extent=[-1, 1.3, -1, 1.3],
                    reduce_C_function=np.sum, xscale='log', yscale='log',
                    norm=weight_norm, linewidths=0.2, cmap='Greys')

    # Plot stellar_data
    im = ax.hexbin(m_age[com_pop], g_hmrs[com_pop], gridsize=50,
                   mincnt=np.min(w) - (0.1 * np.min(w)),
                   C=w[com_pop], extent=[-1, 1.3, -1, 1.3],
                   reduce_C_function=np.sum, xscale='log', yscale='log',
                   norm=weight_norm, linewidths=0.2, cmap='viridis')
    # Plot stellar_data
    im1 = ax.hexbin(m_age[diff_pop], g_hmrs[diff_pop], gridsize=50,
                    mincnt=np.min(w) - (0.1 * np.min(w)),
                    C=w[diff_pop], extent=[-1, 1.3, -1, 1.3],
                    reduce_C_function=np.sum, xscale='log', yscale='log',
                    norm=weight_norm, linewidths=0.2, cmap='Greys')

    # Label axes
    ax.set_ylabel("$R_{\mathrm{gas}} / [\mathrm{pkpc}]$")
    ax.set_xlabel("$R_{\star} / [\mathrm{pkpc}]$")
    ax1.set_xlabel("$Age / [\mathrm{Myr}]$")

    # Remove unused axes
    ax1.tick_params("y", left=False, right=False, labelleft=False,
                    labelright=False)

    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("$\sum w_{i}$")

    # Save figure
    mkdir("plots/stellar_gas_hmr_comp/")
    fig.savefig("plots/stellar_gas_hmr_comp/stellar_gas_hmr_comp_%s.png" % snap,
                bbox_inches="tight")

    plt.close(fig)
