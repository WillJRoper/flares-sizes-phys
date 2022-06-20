from flare import plt as flareplt
import os

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
from scipy.spatial import cKDTree
import numpy as np
import pandas as pd
from utils import calc_3drad, calc_light_mass_rad, mkdir, get_pixel_hlr
import eagle_IO.eagle_IO as eagle_io

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

    # Plot stellar_data
    im = ax.hexbin(mass, hmrs, gridsize=30,
                   mincnt=np.min(w) - (0.1 * np.min(w)),
                   C=w, extent=[-1, 1.3, 8, 11.2],
                   reduce_C_function=np.sum, xscale='log', yscale='log',
                   norm=weight_norm, linewidths=0.2, cmap='viridis')

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
    col = stellar_data["apertures"]["age"][30]

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
    fig = plt.figure(figsize=(3.5, 2 * 3.5))
    gs = gridspec.GridSpec(nrows=2, ncols=1 + 1,
                           width_ratios=[20, ] + [1, ])
    gs.update(wspace=0.0, hspace=0.0)
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])
    ax1 = fig.add_subplot(gs[1, 0])
    cax1 = fig.add_subplot(gs[1, 1])

    # Remove x axis we don't need
    ax.tick_params("x", top=False, bottom=False, labeltop=False,
                   labelbottom=False)

    # Plot stellar_data
    im = ax.hexbin(s_hmrs, g_hmrs, gridsize=30,
                   mincnt=np.min(w) - (0.1 * np.min(w)),
                   C=w, extent=[-1, 1.3, -1, 1.3],
                   reduce_C_function=np.sum, xscale='log', yscale='log',
                   linewidths=0.2, cmap='viridis', norm=weight_norm)
    im1 = ax1.hexbin(s_hmrs, g_hmrs, gridsize=30,
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
    cbar.set_label("$\mathrm{Age} / [\mathrm{Myrs}]$")

    # Save figure
    mkdir("plots/stellar_gas_hmr_comp/")
    fig.savefig("plots/stellar_gas_hmr_comp/stellar_gas_hmr_comp_%s.png" % snap,
                bbox_inches="tight")

    plt.close(fig)


def visualise_gas(stellar_data, gas_data, snap, path):

    # Get redshift
    z = float(snap.split("z")[-1].replace("p", "."))

    # Define softening length
    if z <= 2.8:
        soft = 0.000474390 / 0.6777 * 1e3
    else:
        soft = 0.001802390 / (0.6777 * (1 + z)) * 1e3

    # Define image properties
    res = soft
    width = 60  # pkpc
    ndims = (int(np.ceil(width / res)), int(np.ceil(width / res)))
    width = res * ndims[0]
    imgrange = ((-width / 2, width / 2), (-width / 2, width / 2))

    # Get galaxy data
    regions = stellar_data["regions"]
    sbegin = stellar_data["begin"]
    nstar = stellar_data["Galaxy,S_Length"]
    gbegin = gas_data["begin"]
    ngas = gas_data["Galaxy,G_Length"]
    s_hmrs = stellar_data["HMRs"]
    g_hmrs = gas_data["HMRs"]
    cops = stellar_data["Galaxy,COP"]
    star_pos = stellar_data["Particle,S_Coordinates"]
    gas_pos = gas_data["Particle,G_Coordinates"]
    star_m = stellar_data["Particle,S_Mass"]
    gas_m = gas_data["Particle,G_Mass"]
    gas_app = gas_data["Particle/Apertures/Gas,30"]

    # Load surroundings data for the first region
    prev_reg = 0
    s_star_pos = eagle_io.read_array('PARTDATA', path.replace("<reg>", "00"),
                                     snap, 'PartType4/Coordinates', noH=True,
                                     physicalUnits=True,
                                     numThreads=8) * 10**3
    s_gas_pos = eagle_io.read_array('PARTDATA', path.replace("<reg>", "00"),
                                    snap, 'PartType0/Coordinates', noH=True,
                                    physicalUnits=True,
                                    numThreads=8) * 10**3
    s_star_m = eagle_io.read_array('PARTDATA', path.replace("<reg>", "00"),
                                   snap, 'PartType4/Mass', noH=True,
                                   physicalUnits=True,
                                   numThreads=8)
    s_gas_m = eagle_io.read_array('PARTDATA', path.replace("<reg>", "00"),
                                  snap, 'PartType0/Mass', noH=True,
                                  physicalUnits=True,
                                  numThreads=8)

    # Build kd-tree
    gas_tree = cKDTree(s_gas_pos)

    # Define the two populations
    both_compact = np.logical_and(s_hmrs < 1, g_hmrs < 1)
    extend_gas = np.logical_and(s_hmrs < 1, g_hmrs >= 1)

    # Initialise images
    compgal_img = np.zeros(ndims)
    exgal_img = np.zeros(ndims)
    comp_img = np.zeros(ndims)
    ex_img = np.zeros(ndims)

    # Loop over galaxies in the both compact population
    for (ind, b), l in zip(enumerate(gbegin),
                           ngas):

        if not both_compact[ind]:
            continue

        # Get galaxy positions and centre on COP
        this_gas_pos = gas_pos[b: b + l,
                               :][gas_app[b: b + l], :] - cops[ind, :]

        # Get the particle masses
        this_gas_m = gas_m[b: b + l][gas_app[b: b + l]]

        # Create image of particles in galaxy
        H_gal, _, _, = np.histogram2d(this_gas_pos[:, 0], this_gas_pos[:, 1],
                                      bins=ndims, range=imgrange,
                                      weights=this_gas_m)

        # Get this galaxy's region
        reg = regions[ind]

        # Load the new regions data if we have a new region
        if reg != prev_reg:
            print("Moving on to region:", reg)
            prev_reg = reg
            s_star_pos = eagle_io.read_array('PARTDATA',
                                             path.replace("<reg>",
                                                          str(reg).zfill(2)),
                                             snap, 'PartType4/Coordinates',
                                             noH=True,
                                             physicalUnits=True,
                                             numThreads=8) * 10**3
            s_gas_pos = eagle_io.read_array('PARTDATA',
                                            path.replace("<reg>",
                                                         str(reg).zfill(2)),
                                            snap, 'PartType0/Coordinates',
                                            noH=True,
                                            physicalUnits=True,
                                            numThreads=8) * 10**3
            s_star_m = eagle_io.read_array('PARTDATA',
                                           path.replace("<reg>",
                                                        str(reg).zfill(2)),
                                           snap, 'PartType4/Mass', noH=True,
                                           physicalUnits=True,
                                           numThreads=8)
            s_gas_m = eagle_io.read_array('PARTDATA',
                                          path.replace("<reg>",
                                                       str(reg).zfill(2)),
                                          snap, 'PartType0/Mass', noH=True,
                                          physicalUnits=True,
                                          numThreads=8)

            # Build kd-tree
            gas_tree = cKDTree(s_gas_pos)

        # Get particles we need
        inds = gas_tree.query_ball_point(cops[ind, :], r=width / 2,
                                         workers=8)

        # Get surrounding particle positions and masses
        this_s_gas_pos = s_gas_pos[inds, :] - cops[ind, :]
        this_s_gas_m = s_gas_m[inds]

        # Make the image of the surrounding gas
        H_s, _, _ = np.histogram2d(this_s_gas_pos[:, 0], this_s_gas_pos[:, 1],
                                   bins=ndims, range=imgrange,
                                   weights=this_s_gas_m)

        # Stack images
        compgal_img += H_gal
        comp_img += H_s

    # Loop over galaxies in the extended gas population
    for (ind, b), l in zip(enumerate(gbegin), ngas):

        if not extend_gas[ind]:
            continue

        # Get galaxy positions and centre on COP
        this_gas_pos = gas_pos[b: b + l,
                               :][gas_app[b: b + l], :] - cops[ind, :]

        # Get the particle masses
        this_gas_m = gas_m[b: b + l][gas_app[b: b + l]]

        # Create image of particles in galaxy
        H_gal, _, _, = np.histogram2d(this_gas_pos[:, 0], this_gas_pos[:, 1],
                                      bins=ndims, range=imgrange,
                                      weights=this_gas_m)

        # Get this galaxy's region
        reg = regions[ind]

        # Load the new regions data if we have a new region
        if reg != prev_reg:
            print("Moving on to region:", reg)
            prev_reg = reg
            s_star_pos = eagle_io.read_array('PARTDATA',
                                             path.replace("<reg>",
                                                          str(reg).zfill(2)),
                                             snap, 'PartType4/Coordinates',
                                             noH=True,
                                             physicalUnits=True,
                                             numThreads=8) * 10**3
            s_gas_pos = eagle_io.read_array('PARTDATA',
                                            path.replace("<reg>",
                                                         str(reg).zfill(2)),
                                            snap, 'PartType0/Coordinates',
                                            noH=True,
                                            physicalUnits=True,
                                            numThreads=8) * 10**3
            s_star_m = eagle_io.read_array('PARTDATA',
                                           path.replace("<reg>",
                                                        str(reg).zfill(2)),
                                           snap, 'PartType4/Mass', noH=True,
                                           physicalUnits=True,
                                           numThreads=8)
            s_gas_m = eagle_io.read_array('PARTDATA',
                                          path.replace("<reg>",
                                                       str(reg).zfill(2)),
                                          snap, 'PartType0/Mass', noH=True,
                                          physicalUnits=True,
                                          numThreads=8)

            # Build kd-tree
            gas_tree = cKDTree(s_gas_pos)

        # Get particles we need
        inds = gas_tree.query_ball_point(cops[ind, :], r=width / 2,
                                         workers=8)

        # Get surrounding particle positions and masses
        this_s_gas_pos = s_gas_pos[inds, :] - cops[ind, :]
        this_s_gas_m = s_gas_m[inds]

        # Make the image of the surrounding gas
        H_s, _, _ = np.histogram2d(this_s_gas_pos[:, 0], this_s_gas_pos[:, 1],
                                   bins=ndims, range=imgrange,
                                   weights=this_s_gas_m)

        # Stack images
        exgal_img += H_gal
        ex_img += H_s

    # Calculate image "half mass radii"
    pix_area = res * res
    exgal_hmr = get_pixel_hlr(exgal_img, pix_area, radii_frac=0.5)
    ex_hmr = get_pixel_hlr(ex_img, pix_area, radii_frac=0.5)
    compgal_hmr = get_pixel_hlr(compgal_img, pix_area, radii_frac=0.5)
    comp_hmr = get_pixel_hlr(comp_img, pix_area, radii_frac=0.5)

    # Set up plot
    fig = plt.figure(figsize=(3.5 * 2, 3.5 * 2), dpi=ndims[0])
    gs = gridspec.GridSpec(nrows=2, ncols=2)
    gs.update(wspace=0.0, hspace=0.0)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    # Remove all ticks
    for ax in [ax1, ax2, ax3, ax4]:
        ax.tick_params("both", left=False, right=False, top=False,
                       bottom=False, labelleft=False, labelright=False,
                       labeltop=False, labelbottom=False)
        ax.grid(False)

    # Plot the images
    extent = [-width / 2, width / 2, -width / 2, width / 2]
    im1 = ax1.imshow(compgal_img, cmap="Greys_r",
                     norm=LogNorm(vmin=10**-2.5, vmax=10**2, clip=True),
                     extent=extent)
    im2 = ax2.imshow(exgal_img, cmap="Greys_r",
                     norm=LogNorm(vmin=10**-2.5, vmax=10**2, clip=True),
                     extent=extent)
    im3 = ax3.imshow(comp_img, cmap="Greys_r",
                     norm=LogNorm(vmin=10**-2.5, vmax=10**2, clip=True),
                     extent=extent)
    im4 = ax4.imshow(ex_img, cmap="Greys_r",
                     norm=LogNorm(vmin=10**-2.5, vmax=10**2, clip=True),
                     extent=extent)

    # Label this image grid
    ax1.set_title("Compact Gas")
    ax2.set_title("Extended Gas")
    ax1.set_ylabel("Galaxy")
    ax3.set_ylabel("Surroundings")

    # Save figure
    mkdir("plots/images/")
    fig.savefig("plots/images/gas_dist_stack_%s.png" % snap,
                bbox_inches="tight")
