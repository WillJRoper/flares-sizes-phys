from flare import plt as flareplt
import os

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
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


def plot_stellar_hmr(stellar_data, snap, weight_norm, cut_on="hmr"):

    # Define arrays to store computations
    hmrs = stellar_data["HMRs"]
    mass = stellar_data["mass"]
    den_hmr = stellar_data["apertures"]["density"][cut_on]
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
    com_pop = den_hmr >= stellar_data["density_cut"]
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
    g_hmrs = gas_data["HMRs"]
    w = stellar_data["weight"]
    s_den_hmr = stellar_data["apertures"]["density"]["hmr"]
    col = gas_data["apertures"]["density"]["hmr"]

    # Remove galaxies without stars
    okinds = np.logical_and(g_hmrs > 0, s_hmrs > 0)
    print("Galaxies before spurious cut: %d" % s_hmrs.size)
    s_hmrs = s_hmrs[okinds]
    g_hmrs = g_hmrs[okinds]
    s_den_hmr = s_den_hmr[okinds]
    col = col[okinds]
    w = w[okinds]
    print("Galaxies after spurious cut: %d" % s_hmrs.size)

    # Set up plot
    fig = plt.figure(figsize=(2.75, 2.75))
    gs = gridspec.GridSpec(nrows=2, ncols=ncols + 1,
                           width_ratios=[20, ] * ncols + [1, ])
    gs.update(wspace=0.0, hspace=0.0)
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])
    ax1 = fig.add_subplot(gs[1, 0])
    cax1 = fig.add_subplot(gs[1, 1])

    # Remove x axis we don't need
    ax.tick_params("x", top=False, bottom=False, labeltop=False,
                   labelbottom=False)

    # Plot stellar_data
    im = ax.hexbin(s_hmrs, g_hmrs, gridsize=50,
                   mincnt=np.min(w) - (0.1 * np.min(w)),
                   C=w, extent=[-1, 1.3, -1, 1.3],
                   reduce_C_function=np.mean, xscale='log', yscale='log',
                   linewidths=0.2, cmap='viridis')
    im1 = ax1.hexbin(s_hmrs, g_hmrs, gridsize=50,
                     mincnt=np.min(w) - (0.1 * np.min(w)),
                     C=col, extent=[-1, 1.3, -1, 1.3],
                     reduce_C_function=np.mean, xscale='log', yscale='log',
                     linewidths=0.2, cmap='magma')

    # Set axes y lims
    ax.set_ylim(10**-1.1, 10**1.5)
    ax.set_xlim(10**-1.1, 10**1.5)
    ax1.set_ylim(10**-1.1, 10**1.5)
    ax1.set_xlim(10**-1.1, 10**1.5)

    # Label axes
    ax.set_ylabel("$R_{\mathrm{gas}} / [\mathrm{pkpc}]$")
    ax1.set_ylabel("$R_{\mathrm{gas}} / [\mathrm{pkpc}]$")
    ax1.set_xlabel("$R_{\star} / [\mathrm{pkpc}]$")

    cbar = fig.colorbar(im, cax)
    cbar.set_label("$\sum w_{i}$")

    cbar = fig.colorbar(im1, cax1)
    cbar.set_label("$n_H / [\mathrm{cm}^{-3}]$")

    # Save figure
    mkdir("plots/stellar_gas_hmr_comp/")
    fig.savefig("plots/stellar_gas_hmr_comp/stellar_gas_hmr_comp_%s.png" % snap,
                bbox_inches="tight")

    plt.close(fig)
