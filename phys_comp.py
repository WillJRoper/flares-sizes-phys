import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
import eagle_IO.eagle_IO as eagle_io
from flare import plt as flareplt
from unyt import mh, cm, Gyr, g, Msun, Mpc
from utils import mkdir, plot_meidan_stat, get_nonmaster_evo_data
from utils import get_nonmaster_centred_data, grav_tree
import astropy.units as u
import astropy.constants as const
from astropy.cosmology import Planck18 as cosmo, z_at_value
from scipy.spatial import cKDTree


def plot_birth_density_evo():

    flares_z_bins = np.arange(4.5, 15.5, 1.0)

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

    flares_z_bins = np.arange(4.5, 15.5, 1.0)

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


def plot_sfr_evo_comp(snap):

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

    # Define z bins
    flares_age_bins = np.arange(cosmo.age(5).value, cosmo.age(30).value, -0.1)
    flares_z_bins = [5, ]
    for age in flares_age_bins:
        flares_z_bins.append(z_at_value(cosmo.age,
                                        age * u.Gyr,
                                        zmin=0, zmax=50))

    # Set up the plot
    fig = plt.figure(figsize=(3.5, 3.5))
    ax = fig.add_subplot(111)

    # Log the y axis
    ax.loglog()

    # Set up the plot
    fig = plt.figure(figsize=(3.5, 3.5))
    ax = fig.add_subplot(111)
    ax.semilogy()

    # Loop over the variants
    for t, l, ls in zip(types, labels, linestyles):

        # Get data
        zs, ms = get_nonmaster_evo_data(
            path.replace("<type>", t), snap, y_key="PartType4/InitialMass")

        # Loop over reshift bins
        sfrs = []
        plt_zs = []
        for z_low in flares_z_bins[:-1]:

            z_high = z_at_value(cosmo.age, cosmo.age(z_low) - (100 * u.Myr),
                                zmin=0, zmax=50)

            zokinds = np.logical_and(zs < z_high, zs >= z_low)

            sfrs.append(np.sum(ms[zokinds]) * 10**10 / 100)  # M_sun / Myr
            plt_zs.append(z_low)

        ax.plot(plt_zs, sfrs, label=l, ls=ls)

    ax.set_ylabel(
        r"$\mathrm{SFR}_{100} / [\mathrm{M}_\odot\mathrm{Myr}^{-1}]$")
    ax.set_xlabel(r"$z$")

    ax.legend(loc='upper center',
              bbox_to_anchor=(0.5, -0.2),
              fancybox=True, ncol=3)

    # Save figure
    mkdir("plots/physics_vary/")
    fig.savefig("plots/physics_vary/sfr_evo.png",
                bbox_inches="tight")


def plot_hmr_phys_comp_grid(snap):

    mass_bins = np.logspace(8.0, 13, 30)
    mass_lims = [(10**7.8, 10**11.5), (10**7.8, 10**12.5),
                 (10**7.8, 10**11.2), (10**7.8, 10**12.5)]
    hmr_lims = [(10**0, 10**2), (10**0, 10**2), (10**-0.8, 10**1.3)]

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

    # Define plot grid shape
    nrows = 3
    ncols = 3

    # Set up plot
    fig = plt.figure(figsize=(3.5 * ncols, 3.5 * nrows))
    gs = gridspec.GridSpec(nrows=nrows, ncols=ncols)
    gs.update(wspace=0.0, hspace=0.0)
    axes = np.empty((nrows, ncols), dtype=object)
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

    for i in range(axes.shape[0]):

        if i == 0 or i == 1:
            idata = i
        elif i == 2:
            idata = 4

        for j in range(axes.shape[1]):

            if j == 3:
                jdata = "tot"
            elif j == 0 or j == 1:
                jdata = j
            elif j == 2:
                jdata = 4

            # Loop over the variants
            for t, l, ls in zip(types, labels, linestyles):

                print(i, j, t, l)

                # Get the number stars in a galaxy to perform nstar cut
                nparts = eagle_io.read_array('SUBFIND',
                                             path.replace("<type>", t),
                                             snap,
                                             'Subhalo/SubLengthType',
                                             noH=True, physicalUnits=True,
                                             numThreads=8)
                okinds = np.logical_and(nparts[:, 4] > 0, nparts[:, 0] > 0)
                okinds = np.logical_and(okinds, nparts[:, 1] > 0)

                # Get the arrays from the raw data files
                hmr = eagle_io.read_array('SUBFIND', path.replace("<type>", t),
                                          snap,
                                          'Subhalo/HalfMassRad',
                                          noH=True, physicalUnits=True,
                                          numThreads=8)[:, idata] * 1000
                if jdata == "tot":
                    mass_star = eagle_io.read_array(
                        "SUBFIND",
                        path.replace("<type>", t),
                        snap,
                        "Subhalo/ApertureMeasurements/Mass/030kpc",
                        noH=True, physicalUnits=True,
                        numThreads=8
                    )[:, 4] * 10 ** 10
                    mass_gas = eagle_io.read_array(
                        "SUBFIND",
                        path.replace("<type>", t),
                        snap,
                        "Subhalo/ApertureMeasurements/Mass/030kpc",
                        noH=True, physicalUnits=True,
                        numThreads=8
                    )[:, 0] * 10 ** 10
                    mass_dm = eagle_io.read_array(
                        "SUBFIND",
                        path.replace("<type>", t),
                        snap,
                        "Subhalo/ApertureMeasurements/Mass/030kpc",
                        noH=True, physicalUnits=True,
                        numThreads=8
                    )[:, 1] * 10 ** 10
                    mass_bh = eagle_io.read_array(
                        "SUBFIND",
                        path.replace("<type>", t),
                        snap,
                        "Subhalo/ApertureMeasurements/Mass/030kpc",
                        noH=True, physicalUnits=True,
                        numThreads=8
                    )[:, 5] * 10 ** 10
                    mass = mass_star + mass_dm + mass_gas + mass_bh
                else:
                    mass = eagle_io.read_array(
                        "SUBFIND",
                        path.replace("<type>", t),
                        snap,
                        "Subhalo/ApertureMeasurements/Mass/030kpc",
                        noH=True, physicalUnits=True,
                        numThreads=8
                    )[:, jdata] * 10 ** 10

                # Plot median curves
                # okinds = mass > 0
                plot_meidan_stat(mass[okinds], hmr[okinds],
                                 np.ones(hmr[okinds].size),
                                 axes[i, j], lab=l,
                                 color=None, bins=mass_bins, ls=ls)

    # Label axes
    subscripts = ["\mathrm{gas}", "\mathrm{DM}", "\star", "\mathrm{tot}"]
    for ind, ax in enumerate(axes[:, 0]):
        ax.set_ylabel(r"$R_{1/2, %s}$" % subscripts[ind])
    for ind, ax in enumerate(axes[-1, :]):
        ax.set_xlabel(r"$M_{%s} / M_\odot$" % subscripts[ind])

    # Set axis limits
    for i in range(axes.shape[0]):
        for j in range(axes.shape[1]):
            axes[i, j].set_ylim(hmr_lims[i])
            axes[i, j].set_xlim(mass_lims[j])

    axes[-1, 1].legend(loc='upper center',
                       bbox_to_anchor=(0.5, -0.2),
                       fancybox=True, ncol=7)

    # Save figure
    mkdir("plots/physics_vary/")
    fig.savefig("plots/physics_vary/hmr_grid_%s.png" % snap,
                bbox_inches="tight")


def plot_hmr_phys_comp_grid_1kpc(snap):

    mass_bins = np.logspace(6.0, 11, 30)
    mass_lims = [(10**6.0, 10**11), (10**6.0, 10**11),
                 (10**6.0, 10**11), (10**6.0, 10**11)]
    hmr_lims = [10**-0.8, 10**2]

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

    # Define plot grid shape
    nrows = 3
    ncols = 4

    # Set up plot
    fig = plt.figure(figsize=(3.5 * ncols, 3.5 * nrows))
    gs = gridspec.GridSpec(nrows=nrows, ncols=ncols)
    gs.update(wspace=0.0, hspace=0.0)
    axes = np.empty((nrows, ncols), dtype=object)
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

    for i in range(axes.shape[0]):

        if i == 0 or i == 1:
            idata = i
        elif i == 2:
            idata = 4

        for j in range(axes.shape[1]):

            if j == 3:
                jdata = "tot"
            elif j == 0 or j == 1:
                jdata = j
            elif j == 2:
                jdata = 4

            # Loop over the variants
            for t, l, ls in zip(types, labels, linestyles):

                print(i, j, t, l)

                # Get the number stars in a galaxy to perform nstar cut
                nparts = eagle_io.read_array('SUBFIND',
                                             path.replace("<type>", t),
                                             snap,
                                             'Subhalo/SubLengthType',
                                             noH=True, physicalUnits=True,
                                             numThreads=8)
                okinds = np.logical_and(nparts[:, 4] > 0, nparts[:, 0] > 0)
                okinds = np.logical_and(okinds, nparts[:, 1] > 0)

                # Get the arrays from the raw data files
                hmr = eagle_io.read_array('SUBFIND', path.replace("<type>", t),
                                          snap,
                                          'Subhalo/HalfMassRad',
                                          noH=True, physicalUnits=True,
                                          numThreads=8)[:, idata] * 1000
                if jdata == "tot":
                    mass_star = eagle_io.read_array(
                        "SUBFIND",
                        path.replace("<type>", t),
                        snap,
                        "Subhalo/ApertureMeasurements/Mass/001kpc",
                        noH=True, physicalUnits=True,
                        numThreads=8
                    )[:, 4] * 10 ** 10
                    mass_gas = eagle_io.read_array(
                        "SUBFIND",
                        path.replace("<type>", t),
                        snap,
                        "Subhalo/ApertureMeasurements/Mass/001kpc",
                        noH=True, physicalUnits=True,
                        numThreads=8
                    )[:, 0] * 10 ** 10
                    mass_dm = eagle_io.read_array(
                        "SUBFIND",
                        path.replace("<type>", t),
                        snap,
                        "Subhalo/ApertureMeasurements/Mass/001kpc",
                        noH=True, physicalUnits=True,
                        numThreads=8
                    )[:, 1] * 10 ** 10
                    mass_bh = eagle_io.read_array(
                        "SUBFIND",
                        path.replace("<type>", t),
                        snap,
                        "Subhalo/ApertureMeasurements/Mass/001kpc",
                        noH=True, physicalUnits=True,
                        numThreads=8
                    )[:, 5] * 10 ** 10
                    mass = mass_star + mass_dm + mass_gas + mass_bh
                else:
                    mass = eagle_io.read_array(
                        "SUBFIND",
                        path.replace("<type>", t),
                        snap,
                        "Subhalo/ApertureMeasurements/Mass/001kpc",
                        noH=True, physicalUnits=True,
                        numThreads=8
                    )[:, jdata] * 10 ** 10

                # Plot median curves
                # okinds = mass > 0
                plot_meidan_stat(mass[okinds], hmr[okinds],
                                 np.ones(hmr[okinds].size),
                                 axes[i, j], lab=l,
                                 color=None, bins=mass_bins, ls=ls)

    # Label axes
    subscripts = ["\mathrm{gas}", "\mathrm{DM}", "\star", "\mathrm{tot}"]
    for ind, ax in enumerate(axes[:, 0]):
        ax.set_ylabel(r"$R_{1/2, %s}$" % subscripts[ind])
    for ind, ax in enumerate(axes[-1, :]):
        ax.set_xlabel(r"$M_{%s} / M_\odot$" % subscripts[ind])

    # Set axis limits
    for i in range(axes.shape[0]):
        for j in range(axes.shape[1]):
            axes[i, j].set_ylim(hmr_lims)
            axes[i, j].set_xlim(mass_lims[j])

    axes[-1, 1].legend(loc='upper center',
                       bbox_to_anchor=(1.0, -0.2),
                       fancybox=True, ncol=7)

    # Save figure
    mkdir("plots/physics_vary/")
    fig.savefig("plots/physics_vary/hmr_grid_1kpc_%s.png" % snap,
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


def plot_potential(snap):

    # Get redshift
    z = float(snap.split("z")[-1].replace("p", "."))

    # Define softening length
    if z <= 2.8:
        soft = 0.000474390 / 0.6777
    else:
        soft = 0.001802390 / (0.6777 * (1 + z))

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

    star_keys = ["Mass", "InitialMass", "Feedback_EnergyFraction", ]
    keys = ["Mass", ]

    # Physical constants
    G = (const.G.to(u.km ** 3 * u.M_sun ** -1 * u.s ** -2)).value

    # Set up the plot
    fig = plt.figure(figsize=(3.5, 3.5))
    ax = fig.add_subplot(111)

    # Log the y axis
    ax.loglog()

    # Loop over the variants
    for t, l, ls in zip(types, labels, linestyles):

        print(t, l, ls, snap)

        # Get data dict from the raw data files
        star_data = get_nonmaster_centred_data(path.replace("<type>", t),
                                               snap, star_keys,
                                               part_type=4)
        gas_data = get_nonmaster_centred_data(path.replace("<type>", t),
                                              snap, keys,
                                              part_type=0)
        dm_data = get_nonmaster_centred_data(path.replace("<type>", t),
                                             snap, keys,
                                             part_type=1)
        bh_data = get_nonmaster_centred_data(path.replace("<type>", t),
                                             snap, keys,
                                             part_type=5)

        # Loop over galaxies
        masses = []
        binding_energy = []
        feedback_energy = []
        for key in star_data:
            print(key, len(star_data[key]["PartType4/Mass"])
                  + len(dm_data[key]["PartType1/Mass"]) +
                  len(gas_data[key]["PartType0/Mass"]))

            # Get hmrs
            hmr = star_data[key]["HMR"]

            # Combine particle arrays
            try:
                pos = np.concatenate((star_data[key]["PartType4/Coordinates"],
                                      dm_data[key]["PartType1/Coordinates"],
                                      gas_data[key]["PartType0/Coordinates"],
                                      bh_data[key]["PartType5/Coordinates"]))
                ms = np.concatenate((star_data[key]["PartType4/Mass"],
                                     dm_data[key]["PartType1/Mass"],
                                     gas_data[key]["PartType0/Mass"],
                                     bh_data[key]["PartType5/Mass"])) * 10 ** 10
            except KeyError:
                pos = np.concatenate((star_data[key]["PartType4/Coordinates"],
                                      dm_data[key]["PartType1/Coordinates"],
                                      gas_data[key]["PartType0/Coordinates"]))
                ms = np.concatenate((star_data[key]["PartType4/Mass"],
                                     dm_data[key]["PartType1/Mass"],
                                     gas_data[key]["PartType0/Mass"])) * 10 ** 10

            gas_pos = np.array(gas_data[key]["PartType0/Coordinates"])
            gas_ms = np.array(gas_data[key]["PartType0/Mass"]) * 10 ** 10

            # Calculate feedback energy
            zs = star_data[key]['PartType4/StellarFormationTime']
            z_low = z_at_value(cosmo.age, cosmo.age(z) - (0.03 * u.Gyr),
                               zmin=0, zmax=50)
            z_high = z_at_value(cosmo.age, cosmo.age(z) - (0.06 * u.Gyr),
                                zmin=0, zmax=50)
            zokinds = np.logical_and(zs < z_high, zs >= z_low)
            ini_ms = np.array(
                star_data[key]['PartType4/InitialMass'])[zokinds] * 10 ** 10

            if ini_ms.size == 0:
                continue

            fths = np.array(star_data[key][
                'PartType4/Feedback_EnergyFraction'])[zokinds]
            star_pos = np.array(star_data[key][
                "PartType4/Coordinates"])[zokinds]

            # Build tree
            all_tree = cKDTree(pos)
            gas_tree = cKDTree(gas_pos)

            for (ind, m), fth in zip(enumerate(ini_ms), fths):

                # Find neighbouring gas particles to this stellar particle
                dist, inds = gas_tree.query(star_pos[ind, :],
                                            k=int(np.ceil(1.3 * fth)))
                print(inds)

                # Calculate binding energy
                e_bind = grav_tree(all_tree, gas_pos[inds, :],
                                   soft, ms, gas_ms[inds], z,
                                   G).to(u.erg).value
                if type(e_bind) is np.float64:
                    binding_energy.append(e_bind)
                else:
                    binding_energy.extend(e_bind)

                # Calculate feedback from this particle
                for i in range(len(inds)):
                    feedback_energy.append(1.74 * 10 ** (49 - 50) * m)

                    # Get galaxy mass
                    masses.append(star_data[key]["Mass"] * 10 ** 10)

        masses = np.array(masses)
        binding_energy = np.array(binding_energy)
        feedback_energy = np.array(feedback_energy)

        print(masses, len(masses))
        print(binding_energy, len(binding_energy))
        print(feedback_energy, len(feedback_energy))

        # Plot median curves
        plot_meidan_stat(masses, binding_energy / feedback_energy,
                         np.ones(masses.size),
                         ax, lab=l, bins=mass_bins, color=None, ls=ls)

    # Label axes
    ax.set_ylabel(r"$E_{\mathrm{grav} / E_{\mathrm{FB}}$")
    ax.set_xlabel(r"$M_{\star} / M_\odot$")

    ax.legend(loc='upper center',
              bbox_to_anchor=(0.5, -0.2),
              fancybox=True, ncol=3)

    # Save figure
    mkdir("plots/physics_vary/")
    fig.savefig("plots/physics_vary/potential_energy_%s.png" % snap,
                bbox_inches="tight")
