from flare import plt as flareplt
import os

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from utils import calc_3drad, calc_light_mass_rad, mkdir, plot_meidan_stat

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


def plot_stellar_density(stellar_data, snap, weight_norm):
    # Define x and y limits
    hmrlims = (-1.3, 1.3)
    mlims = (8, 11)
    denlims = (3, 13.5)

    # Define arrays to store computations
    hmrs = np.zeros(len(stellar_data["begin"]))
    mass = np.zeros(len(stellar_data["begin"]))
    den = {key: np.zeros(len(stellar_data["begin"]))
           for key in [0.05, 0.1, 0.5, 1]}
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

        if l < 100:
            continue

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
        axes[i + 1].hexbin(hmrs[okinds], den[r][okinds], gridsize=50,
                           mincnt=np.min(w[okinds]) - (0.1 * np.min(w)),
                           C=w[okinds],
                           extent=[hmrlims[0], hmrlims[1], denlims[0],
                                   denlims[1]],
                           reduce_C_function=np.sum, xscale='log',
                           yscale='log',
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
        axes[i + 1].set_title("R=%.2f pkpc" % r)

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
    axes_twin = []
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
        axes[i + 1].hexbin(mass[okinds], den[r][okinds], gridsize=50,
                           mincnt=np.min(w[okinds]) - (0.1 * np.min(w)),
                           C=w[okinds],
                           extent=[mlims[0], mlims[1], denlims[0], denlims[1]],
                           reduce_C_function=np.sum, xscale='log',
                           yscale='log',
                           norm=weight_norm, linewidths=0.2, cmap='viridis')

    p = plot_meidan_stat(mass, den_hmr, w, axes[0], "$R=R_{1/2}$",
                         color=None, bins=None, ls='-')

    # Set lims
    for ax in axes:
        ax.set_ylim(10 ** denlims[0], 10 ** denlims[1])
        ax.set_xlim(10 ** mlims[0], 10 ** mlims[1])

    # Set titles
    axes[0].set_title("$R=R_{1/2}$")
    for i, r in enumerate(den):
        axes[i + 1].set_title("R=%.2f pkpc" % r)

    # Label axes
    axes[0].set_ylabel(r"$\rho_\star(<R) / [M_\odot / \mathrm{pkpc}^3]$")
    for i in range(ncols):
        axes[i].set_xlabel("$M_\star(<R) / M_\odot$")

    cbar = fig.colorbar(im, cax)
    cbar.set_label("$\sum w_{i}$")

    # Save figure
    mkdir("plots/density/")
    fig.savefig("plots/density/stellar_density_mass_%s.png" % snap,
                bbox_inches="tight")

    plt.close(fig)


def plot_stellar_density_grid(stellar_data, snap, weight_norm):
    # Define x and y limits
    hmrlims = (-1.3, 1.3)
    mlims = (8, 11)
    denlims = (3, 13.5)
    age_lims = (0, 500)
    met_lims = (0, 2)

    # Define arrays to store computations
    hmrs = np.zeros(len(stellar_data["begin"]))
    mass = np.zeros(len(stellar_data["begin"]))
    den_hmr = np.zeros(len(stellar_data["begin"]))
    mass_hmr = np.zeros(len(stellar_data["begin"]))
    ages_hmr = np.zeros(len(stellar_data["begin"]))
    met_hmr = np.zeros(len(stellar_data["begin"]))
    mass_r = {key: np.zeros(len(stellar_data["begin"]))
              for key in [0.05, 0.1, 0.5, 1]}
    ages_r = {key: np.zeros(len(stellar_data["begin"]))
              for key in [0.05, 0.1, 0.5, 1]}
    met_r = {key: np.zeros(len(stellar_data["begin"]))
             for key in [0.05, 0.1, 0.5, 1]}
    den = {key: np.zeros(len(stellar_data["begin"]))
           for key in [0.05, 0.1, 0.5, 1]}
    w = np.zeros(len(stellar_data["begin"]))

    # Loop over galaxies and calculate stellar HMR and denisty within HMR
    for (igal, b), l in zip(enumerate(stellar_data["begin"]),
                            stellar_data["Galaxy,S_Length"]):

        if l < 100:
            continue

        # Get this galaxy's stellar_data
        app = stellar_data["Particle/Apertures/Star,30"][b: b + l]
        cop = stellar_data["Galaxy,COP"][igal]
        pos = stellar_data["Particle,S_Coordinates"][b: b + l, :][app]
        ini_ms = stellar_data["Particle,S_MassInitial"][b: b + l][app]
        ms = stellar_data["Particle,S_Mass"][b: b + l][app]
        ages = stellar_data["Particle,S_Age"][b: b + l][app]
        mets = stellar_data["Particle,S_Z_smooth"][b: b + l][app]

        # Compute particle radii
        rs = calc_3drad(pos - cop)

        # Compute HMR
        hmr = calc_light_mass_rad(rs, ms, radii_frac=0.5)

        # Store results
        den_hmr[igal] = (np.sum(ms[rs <= hmr]) * 10 ** 10
                         / (4 / 3 * np.pi * hmr ** 3))
        mass_hmr[igal] = np.sum(ms[rs <= hmr]) * 10 ** 10
        ages_hmr[igal] = np.average(ages[rs < hmr], weights=ini_ms[rs < hmr])
        met_hmr[igal] = np.average(mets[rs < hmr], weights=ini_ms[rs < hmr])
        hmrs[igal] = hmr
        mass[igal] = np.sum(ms) * 10 ** 10
        w[igal] = stellar_data["weights"][igal]

        for r in [0.05, 0.1, 0.5, 1]:
            mass_r[r][igal] = np.sum(ms[rs <= r]) * 10 ** 10
            ages_r[r][igal] = np.average(ages[rs < r], weights=ini_ms[rs < r])
            met_r[r][igal] = np.average(mets[rs < r], weights=ini_ms[rs < r])
            den[r][igal] = (mass_r[r][igal] / (4 / 3 * np.pi * r ** 3))


    # Define how mnay columns
    nrows = 1 + len(den)
    ncols = 5

    # Set up plot
    fig = plt.figure(figsize=(2.25 * ncols, 2.25 * nrows))
    gs = gridspec.GridSpec(nrows=nrows, ncols=ncols + 1,
                           width_ratios=[20, ] * ncols + [1, ])
    gs.update(wspace=0.0, hspace=0.0)
    axes = np.empty((nrows, ncols), dtype=object)
    cax = fig.add_subplot(gs[:, -1])
    i = 0
    while i < nrows:
        j = 0
        while j < ncols:
            axes[i, j] = fig.add_subplot(gs[i, j])
            axes[i, j].loglog()
            if j > 0:
                axes[i, j].tick_params(axis='y', left=False, right=False,
                                       labelleft=False, labelright=False)
            if i < nrows - 1:
                axes[i, j].tick_params(axis='x', top=False, bottom=False,
                                       labeltop=False, labelbottom=False)
            j += 1
        i += 1

    # Plot stellar_data
    for j, (x, x_ex) in enumerate(zip([mass, mass_hmr, ages_hmr, met_hmr, hmrs],
                                      [mlims, mlims, age_lims, met_lims, hmrlims]
                                     )):

        # Define Boolean indices to remove anomalous results
        okinds = np.logical_and(x > 0, den_hmr > 0)
        
        im = axes[0, j].hexbin(x[okinds], den_hmr[okinds], gridsize=50,
                               mincnt=np.min(w[okinds]) - (0.1 * np.min(w[okinds])),
                               C=w[okinds],
                               extent=[x_ex[0], x_ex[1],
                                       denlims[0], denlims[1]],
                               reduce_C_function=np.sum, xscale='log',
                               yscale='log',
                               norm=weight_norm, linewidths=0.2,
                               cmap='viridis')

        p = plot_meidan_stat(x[okinds], den_hmr[okinds], w[okinds],
                             axes[0, j], "R=R_{1/2}",
                             color=None, bins=None, ls='--')

        
    # Plot weighted medians
    for i, r in enumerate(den):
        for j, (x, x_ex) in enumerate(zip([mass, mass_r[r], ages_r[r], met_r[r], hmrs],
                                    [mlims, mlims, age_lims, met_lims, hmrlims]
                                    )):

            # Define Boolean indices to remove anomalous results
            okinds = np.logical_and(x > 0, den[r] > 0)
        
            im = axes[i + 1, j].hexbin(x[okinds], den[r][okinds], gridsize=50,
                                       mincnt=np.min(w[okinds]) - (0.1 * np.min(w[okinds])),
                               C=w[okinds],
                               extent=[x_ex[0], x_ex[1],
                                       denlims[0], denlims[1]],
                               reduce_C_function=np.sum, xscale='log',
                               yscale='log',
                               norm=weight_norm, linewidths=0.2,
                               cmap='viridis')

            p = plot_meidan_stat(x[okinds], den[r][okinds], w[okinds],
                                 axes[i + 1, j], "R",
                                 color=None, bins=None, s='--')
        
    # Set lims
    for ax in axes:
        ax.set_ylim(10 ** denlims[0], 10 ** denlims[1])
        ax.set_xlim(10 ** hmrlims[0], 10 ** hmrlims[1])

    # Label axes
    for i, lab in enumerate(["HMR", ] + list(den.keys())):
        if type(lab) == str:
            axes[i, 0].set_ylabel(r"$\rho_\star(r<R_{%s}]) / [M_\odot / \mathrm{pkpc}^3]$" % lab)
        else:
            axes[i, 0].set_ylabel(r"$\rho_\star(r<R=%.2f / [pkpc]) / [M_\odot / \mathrm{pkpc}^3]$" % lab)

    axes[-1, 0].set_xlabel("$M_{\star}(r<30 / [pkpc]) / M_\odot$")
    axes[-1, 1].set_xlabel("$M_{\star}(r<R) / M_\odot$")
    axes[-1, 2].set_xlabel("$T(r<R) / [\mathrm{Myr}]$")
    axes[-1, 3].set_xlabel("$Z_\star(r<R)$")
    axes[-1, 4].set_xlabel("$R_{1/2} / [pkpc]$")
    
    cbar = fig.colorbar(im, cax)
    cbar.set_label("$\sum w_{i}$")

    # Save figure
    mkdir("plots/density/")
    fig.savefig("plots/density/stellar_density_grid_hmr_%s.png" % snap,
                bbox_inches="tight")

    plt.close(fig)
