import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
import eagle_IO.eagle_IO as eagle_io
from flare import plt as flareplt
from unyt import mh, cm, Gyr, g, Msun, Mpc
from utils import mkdir, plot_meidan_stat


def plot_birth_density_evo():

    flares_z_bins = np.arange(4.5, 23.5, 1.0)

    # Define the path
    path = "/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/<type>/data/"

    # Define physics variations directories
    types = ["G-EAGLE_00", "FLARES_00_REF", "FLARES_00_highFBlim",
             "FLARES_00_medFBlim", "FLARES_00_slightFBlim",
             "FLARES_00_instantFB", "FLARES_00_noZSFthresh"]

    # Define physics variations directories
    labels = ["AGNdT9", "REF", "$f_{\mathrm{th, max}}=10$",
              "$f_{\mathrm{th, max}}=6$", "$f_{\mathrm{th, max}}=4$",
              "InstantFB", "$Z^0$"]

    # Define linestyles
    linestyles = ["-", "-", "--", "--", "--", "dotted", "dotted"]

    # Define snapshot for the root
    snap = "011_z004p770"

    # Set up the plot
    fig = plt.figure(figsize=(3.5, 3.5))
    ax = fig.add_subplot(111)

    # Log the y axis
    ax.semilogy()

    # Loop over the variants
    for t, l, ls in zip(types, labels, linestyles):

        # Get the arrays from the raw data files
        aborn = eagle_io.read_array('PARTDATA', path.replace("<type>", t),
                                    snap,
                                    'PartType4/StellarFormationTime',
                                    noH=True, physicalUnits=True,
                                    numThreads=8)
        den_born = (eagle_io.read_array("PARTDATA", path.replace("<type>", t),
                                        snap, "PartType4/BirthDensity",
                                        noH=True, physicalUnits=True,
                                        numThreads=8) * 10**10
                    * Msun / Mpc ** 3 / mh).to(1 / cm ** 3).value

        # Convert to redshift
        zs = 1 / aborn - 1

        # Plot median curves
        plot_meidan_stat(zs, den_born, np.ones(den_born.size), ax,
                         lab=l, bins=flares_z_bins, color=None, ls=ls)

    # Label axes
    ax.set_ylabel(r"$n_{\mathrm{H}} / \mathrm{cm}^{-3}$")
    ax.set_xlabel(r"$z_{\mathrm{birth}}$")

    ax.legend(loc='upper center',
              bbox_to_anchor=(0.5, -0.2),
              fancybox=True, ncol=3)

    # Save figure
    mkdir("plots/physics_vary/")
    fig.savefig("plots/physics_vary/stellar_birthden_%s.png" % snap,
                bbox_inches="tight")


def plot_birth_met_evo():

    flares_z_bins = np.arange(4.5, 23.5, 1.0)

    # Define the path
    path = "/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/<type>/data/"

    # Define physics variations directories
    types = ["G-EAGLE_00", "FLARES_00_REF", "FLARES_00_highFBlim",
             "FLARES_00_medFBlim", "FLARES_00_slightFBlim",
             "FLARES_00_instantFB", "FLARES_00_noZSFthresh"]

    # Define labels for each
    labels = ["AGNdT9", "REF", "$f_{\mathrm{th, max}}=10$",
              "$f_{\mathrm{th, max}}=6$", "$f_{\mathrm{th, max}}=4$",
              "InstantFB", "$Z^{0}$"]

    # Define linestyles
    linestyles = ["-", "-", "--", "--", "--", "dotted", "dotted"]

    # Define snapshot for the root
    snap = "011_z004p770"

    # Set up the plot
    fig = plt.figure(figsize=(3.5, 3.5))
    ax = fig.add_subplot(111)

    # Log the y axis
    ax.semilogy()

    # Loop over the variants
    for t, l, ls in zip(types, labels, linestyles):

        # Get the arrays from the raw data files
        aborn = eagle_io.read_array('PARTDATA', path.replace("<type>", t),
                                    snap,
                                    'PartType4/StellarFormationTime',
                                    noH=True, physicalUnits=True,
                                    numThreads=8)
        met = eagle_io.read_array("PARTDATA", path.replace("<type>", t),
                                  snap, "PartType4/SmoothedMetallicity",
                                  noH=True, physicalUnits=True,
                                  numThreads=8)

        # Convert to redshift
        zs = 1 / aborn - 1

        # Plot median curves
        plot_meidan_stat(zs, met, np.ones(met.size), ax,
                         lab=l, color=None, bins=flares_z_bins, ls=ls)

    # Label axes
    ax.set_ylabel(r"$Z_{\mathrm{birth}}$")
    ax.set_xlabel(r"$z_{\mathrm{birth}}$")

    ax.legend(loc='upper center',
              bbox_to_anchor=(0.5, -0.2),
              fancybox=True, ncol=3)

    # Save figure
    mkdir("plots/physics_vary/")
    fig.savefig("plots/physics_vary/stellar_met_%s.png" % snap,
                bbox_inches="tight")


def plot_hmr_phys_comp(snap):

    mass_bins = np.logspace(7.5, 11.5, 30)

    # Define the path
    path = "/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/<type>/data/"

    # Define physics variations directories
    types = ["G-EAGLE_00", "FLARES_00_REF", "FLARES_00_highFBlim",
             "FLARES_00_medFBlim", "FLARES_00_slightFBlim",
             "FLARES_00_instantFB", "FLARES_00_noZSFthresh"]

    # Define labels for each
    labels = ["AGNdT9", "REF", "$f_{\mathrm{th, max}}=10$",
              "$f_{\mathrm{th, max}}=6$", "$f_{\mathrm{th, max}}=4$",
              "InstantFB", "$Z^0$"]

    # Define linestyles
    linestyles = ["-", "-", "--", "--", "--", "dotted", "dotted"]

    # Set up the plot
    fig = plt.figure(figsize=(3.5, 3.5))
    ax = fig.add_subplot(111)

    # Log the y axis
    ax.loglog()

    # Loop over the variants
    for t, l, ls in zip(types, labels, linestyles):

        # Get the arrays from the raw data files
        hmr = eagle_io.read_array('SUBFIND', path.replace("<type>", t),
                                  snap,
                                  'Subhalo/HalfMassRad',
                                  noH=True, physicalUnits=True,
                                  numThreads=8)[:, 4] * 1000
        mass = eagle_io.read_array("SUBFIND", path.replace("<type>", t),
                                   snap,
                                   "Subhalo/ApertureMeasurements/Mass/030kpc",
                                   noH=True, physicalUnits=True,
                                   numThreads=8)[:, 4] * 10 ** 10

        # Plot median curves
        okinds = mass > 0
        plot_meidan_stat(mass[okinds], hmr[okinds], np.ones(hmr[okinds].size),
                         ax, lab=l, color=None, bins=mass_bins, ls=ls)

    # Label axes
    ax.set_ylabel(r"$R_{1/2}$")
    ax.set_xlabel(r"$M_{\star} / M_\odot$")

    ax.legend(loc='upper center',
              bbox_to_anchor=(0.5, -0.2),
              fancybox=True, ncol=3)

    # Save figure
    mkdir("plots/physics_vary/")
    fig.savefig("plots/physics_vary/stellar_hmr_%s.png" % snap,
                bbox_inches="tight")


def plot_gashmr_phys_comp(snap):

    mass_bins = np.logspace(7.5, 11.5, 30)

    # Define the path
    path = "/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/<type>/data/"

    # Define physics variations directories
    types = ["G-EAGLE_00", "FLARES_00_REF", "FLARES_00_highFBlim",
             "FLARES_00_medFBlim", "FLARES_00_slightFBlim",
             "FLARES_00_instantFB", "FLARES_00_noZSFthresh"]

    # Define labels for each
    labels = ["AGNdT9", "REF", "$f_{\mathrm{th, max}}=10$",
              "$f_{\mathrm{th, max}}=6$", "$f_{\mathrm{th, max}}=4$",
              "InstantFB", "$Z^0$"]

    # Define linestyles
    linestyles = ["-", "-", "--", "--", "--", "dotted", "dotted"]

    # Set up the plot
    fig = plt.figure(figsize=(3.5, 3.5))
    ax = fig.add_subplot(111)

    # Log the y axis
    ax.loglog()

    # Loop over the variants
    for t, l, ls in zip(types, labels, linestyles):

        # Get the arrays from the raw data files
        hmr = eagle_io.read_array('SUBFIND', path.replace("<type>", t),
                                  snap,
                                  'Subhalo/HalfMassRad',
                                  noH=True, physicalUnits=True,
                                  numThreads=8)[:, 0] * 1000
        mass = eagle_io.read_array("SUBFIND", path.replace("<type>", t),
                                   snap,
                                   "Subhalo/ApertureMeasurements/Mass/030kpc",
                                   noH=True, physicalUnits=True,
                                   numThreads=8)[:, 4] * 10 ** 10

        # Plot median curves
        okinds = mass > 0
        plot_meidan_stat(mass[okinds], hmr[okinds], np.ones(hmr[okinds].size),
                         ax, lab=l, bins=mass_bins, color=None, ls=ls)

    # Label axes
    ax.set_ylabel(r"$R_{1/2}$")
    ax.set_xlabel(r"$M_{\star} / M_\odot$")

    ax.legend(loc='upper center',
              bbox_to_anchor=(0.5, -0.2),
              fancybox=True, ncol=3)

    # Save figure
    mkdir("plots/physics_vary/")
    fig.savefig("plots/physics_vary/gas_hmr_%s.png" % snap,
                bbox_inches="tight")
