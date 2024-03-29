import os

from flare import plt as flareplt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from utils import calc_3drad, calc_light_mass_rad, mkdir, plot_meidan_stat, age2z
from unyt import mh, cm, Gyr, g, Msun, kpc, Mpc
import eagle_IO.eagle_IO as eagle_io
import pandas as pd

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


def plot_stellar_density_grid(stellar_data, snap, weight_norm):

    # Define redshift
    z = float(snap.split("z")[-1].replace("p", "."))

    # Define x and y limits
    hmrlims = (-1.3, 1.5)
    mlims = (7.8, 12)
    mrlims = (6, 12)
    denlims = (-2, 5)
    age_lims = (0, 3)
    met_lims = (-4.5, -0.9)

    # Define arrays to store computations
    hmrs = stellar_data["HMRs"]
    mass = stellar_data["mass"]
    den_hmr = stellar_data["apertures"]["density"]["hmr"]
    mass_hmr = stellar_data["apertures"]["mass"]["hmr"]
    ages_hmr = stellar_data["apertures"]["age"]["hmr"]
    met_hmr = stellar_data["apertures"]["metal"]["hmr"]
    mass_r = stellar_data["apertures"]["mass"]
    ages_r = stellar_data["apertures"]["age"]
    met_r = stellar_data["apertures"]["metal"]
    den = stellar_data["apertures"]["density"]
    w = stellar_data["weight"]

    # Define how mnay columns
    nrows = 1 + len(den)
    ncols = 4

    # Set up plot
    fig = plt.figure(figsize=(2.5 * ncols, 2.5 * nrows))
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
            if j > 0:
                axes[i, j].tick_params(axis='y', left=False, right=False,
                                       labelleft=False, labelright=False)
            if i < nrows - 1:
                axes[i, j].tick_params(axis='x', top=False, bottom=False,
                                       labeltop=False, labelbottom=False)
            j += 1
        i += 1

    # Set list of xs
    xs = [mass, ages_hmr, met_hmr, hmrs]
    x_exs = [mlims, age_lims, met_lims, hmrlims]

    # Plot stellar_data
    for j, (x, x_ex) in enumerate(zip(xs, x_exs)):

        # Define Boolean indices to remove anomalous results
        okinds = np.logical_and(x > 0, den_hmr > 0)

        if x[okinds].size == 0:
            continue

        im = axes[0, j].hexbin(x[okinds], den_hmr[okinds], gridsize=50,
                               mincnt=np.min(w) - (0.1 * np.min(w)),
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
        # Set xs to loop over
        xs_r = [mass, ages_r[r], met_r[r], hmrs]
        for j, (x, x_ex) in enumerate(zip(xs_r, x_exs)):

            # Define Boolean indices to remove anomalous results
            okinds = np.logical_and(x > 0, den[r] > 0)

            if x[okinds].size == 0:
                continue

            im = axes[i + 1, j].hexbin(x[okinds], den[r][okinds], gridsize=50,
                                       mincnt=np.min(
                w) - (0.1 * np.min(w)),
                C=w[okinds],
                extent=[x_ex[0], x_ex[1],
                        denlims[0], denlims[1]],
                reduce_C_function=np.sum, xscale='log',
                yscale='log',
                norm=weight_norm, linewidths=0.2,
                cmap='viridis')

            p = plot_meidan_stat(x[okinds], den[r][okinds], w[okinds],
                                 axes[i + 1, j], "R",
                                 color=None, bins=None, ls='--')

    # Set lims
    for i in range(axes.shape[0]):
        for j, ex in zip(range(axes.shape[1]),
                         [mlims, age_lims, met_lims, hmrlims]):
            axes[i, j].set_ylim(10 ** denlims[0], 10 ** denlims[1])

            axes[i, j].set_xlim(10 ** ex[0], 10 ** ex[1])

            # Draw line indicating density cut
            axes[i, j].axhline(10**2.5, linestyle="dotted", color="k",
                               alpha=0.8)

    # Label axes
    for i, lab in enumerate(["HMR", ] + list(den.keys())):
        if type(lab) == str:
            axes[i, 0].set_ylabel(
                r"$\rho_\star(<R_{%s}]) / M_\odot\mathrm{pkpc}^{-3}$" % lab)
        else:
            axes[i, 0].set_ylabel(
                r"$\rho_\star(<R_{%.1f}) / M_\odot\mathrm{pkpc}^{-3}$" % lab)

    axes[-1, 0].set_xlabel("$M_{\star}(r<30 / [pkpc]) / M_\odot$")
    axes[-1, 1].set_xlabel("$\mathrm{Age}(r<R) / Myr$")
    axes[-1, 2].set_xlabel("$Z_\star(r<R)$")
    axes[-1, 3].set_xlabel("$R_{1/2} / [pkpc]$")

    cbar = fig.colorbar(im, cax)
    cbar.set_label("$\sum w_{i}$")

    # Save figure
    mkdir("plots/density/")
    fig.savefig("plots/density/stellar_density_grid_hmr_%s.png" % snap,
                bbox_inches="tight")

    plt.close(fig)


def plot_gas_density_mass(regions, snap, path, weight_norm):

    # Load weights
    df = pd.read_csv('../weight_files/weights_grid.txt')
    weights = np.array(df['weights'])

    # Create lists for results
    masses = []
    dens = []
    w = []

    # Loop over regions
    for reg in regions:

        mass = eagle_io.read_array("SUBFIND", path.replace("<reg>", reg),
                                   snap,
                                   "Subhalo/ApertureMeasurements/Mass/030kpc",
                                   noH=True, physicalUnits=True,
                                   numThreads=8)[:, 4] * 10 ** 10
        gal_gmass = eagle_io.read_array("SUBFIND", path.replace("<reg>", reg),
                                        snap,
                                        "Subhalo/ApertureMeasurements/Mass/030kpc",
                                        noH=True, physicalUnits=True,
                                        numThreads=8)[:, 0] * 10 ** 10
        hmrs = eagle_io.read_array("SUBFIND", path.replace("<reg>", reg),
                                   snap,
                                   "Subhalo/HalfMassRad",
                                   noH=True, physicalUnits=True,
                                   numThreads=8)[:, 4] * 10 ** 3
        cops = eagle_io.read_array("SUBFIND", path.replace("<reg>", reg),
                                   snap,
                                   "Subhalo/CentreOfPotential",
                                   noH=True, physicalUnits=True,
                                   numThreads=8) * 1000
        grps = eagle_io.read_array("SUBFIND", path.replace("<reg>", reg),
                                   snap,
                                   "Subhalo/GroupNumber",
                                   noH=True, physicalUnits=True,
                                   numThreads=8)
        subgrps = eagle_io.read_array("SUBFIND", path.replace("<reg>", reg),
                                      snap,
                                      "Subhalo/SubGroupNumber",
                                      noH=True, physicalUnits=True,
                                      numThreads=8)
        g_den = eagle_io.read_array("PARTDATA", path.replace("<reg>", reg),
                                    snap,
                                    "PartType0/Density",
                                    noH=True, physicalUnits=True,
                                    numThreads=8)
        g_mass = eagle_io.read_array("PARTDATA", path.replace("<reg>", reg),
                                     snap,
                                     "PartType0/Mass",
                                     noH=True, physicalUnits=True,
                                     numThreads=8)
        coords = eagle_io.read_array("PARTDATA", path.replace("<reg>", reg),
                                     snap,
                                     "PartType0/Coordinates",
                                     noH=True, physicalUnits=True,
                                     numThreads=8) * 1000
        part_grps = eagle_io.read_array("PARTDATA", path.replace("<reg>", reg),
                                        snap,
                                        "PartType0/GroupNumber",
                                        noH=True, physicalUnits=True,
                                        numThreads=8)
        part_subgrps = eagle_io.read_array("PARTDATA",
                                           path.replace("<reg>", reg),
                                           snap,
                                           "PartType0/SubGroupNumber",
                                           noH=True, physicalUnits=True,
                                           numThreads=8)

        # Apply some cuts
        mokinds = np.logical_and(mass > 10**8, gal_gmass > 10**8)
        mass = mass[mokinds]
        cops = cops[mokinds, :]
        grps = grps[mokinds]
        subgrps = subgrps[mokinds]
        hmrs = hmrs[mokinds]

        # Loop over galaxies
        for igal in range(mass.size):

            # Get galaxy data
            m = mass[igal]
            cop = cops[igal, :]
            g = grps[igal]
            sg = subgrps[igal]
            hmr = hmrs[igal]

            # Get this galaxies stars
            gokinds = np.logical_and(part_grps == g, part_subgrps == sg)
            this_coords = coords[gokinds, :] - cop
            this_den = g_den[gokinds]
            this_gmass = g_mass[gokinds]

            # Compute stellar radii
            rs = np.sqrt(this_coords[:, 0] ** 2
                         + this_coords[:, 1] ** 2
                         + this_coords[:, 2] ** 2)

            # Get only particles within the aperture
            rokinds = rs < 30
            rs = rs[rokinds]
            this_den = this_den[rokinds]
            this_gmass = this_gmass[rokinds]

            # Calculate and store results for this halo
            dens.append(this_gmass * this_den / np.sum(this_gmass))
            masses.append(m)
            w.append(weights[int(reg)])

    # Convert to arrays
    dens = (np.array(dens) * 10**10
            * Msun / Mpc ** 3 / mh).to(1 / cm ** 3).value
    masses = np.array(masses)
    w = np.array(w)

    # Remove spurious objects
    okinds = dens > 0

    # Set up plot
    fig = plt.figure(figsize=(3.5, 3.5))
    ax = fig.add_subplot(111)
    ax.loglog()

    # Plot stellar_data
    im = ax.hexbin(mass[okinds], dens[okinds], gridsize=30,
                   mincnt=np.min(w) - (0.1 * np.min(w)),
                   C=w[okinds], reduce_C_function=np.sum,
                   xscale='log', yscale='log',
                   norm=weight_norm, linewidths=0.2, cmap='viridis')

    # Label axes
    ax.set_xlabel("$M_\star / M_\odot$")
    ax.set_ylabel(r"$n_{\mathrm{H}} / \mathrm{cm}^{-3}$")

    cbar = fig.colorbar(im)
    cbar.set_label("$\sum w_{i}$")

    # Save figure
    mkdir("plots/density/")
    fig.savefig("plots/density/gas_density_mass_relation_%s.png" % snap,
                bbox_inches="tight")

    plt.close(fig)
