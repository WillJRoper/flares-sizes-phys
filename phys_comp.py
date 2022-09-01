import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
import eagle_IO.eagle_IO as eagle_io
from flare import plt as flareplt
from unyt import mh, cm, Gyr, g, Msun, Mpc
from utils import mkdir, plot_meidan_stat, get_nonmaster_evo_data
from utils import get_nonmaster_centred_data, grav_enclosed, calc_ages
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
    G = (const.G.to(u.Mpc ** 3 * u.M_sun ** -1 * u.yr ** -2))

    # Set up the plot
    fig = plt.figure(figsize=(3.5, 3.5))
    ax = fig.add_subplot(111)

    # Log the y axis
    ax.semilogx()

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
            print(key,
                  len(star_data[key]["PartType4/Mass"]),
                  len(dm_data[key]["PartType1/Mass"]),
                  len(gas_data[key]["PartType0/Mass"]))

            # Get hmrs
            hmr = star_data[key]["HMR"]
            cop = star_data[key]["COP"]

            gas_pos = np.array(gas_data[key]["PartType0/Coordinates"])
            dm_pos = np.array(dm_data[key]["PartType1/Coordinates"])
            all_star_pos = np.array(star_data[key]["PartType4/Coordinates"])
            gas_ms = np.array(gas_data[key]["PartType0/Mass"]) * 10 ** 10
            dm_ms = np.array(dm_data[key]["PartType1/Mass"]) * 10 ** 10
            star_ms = np.array(star_data[key]["PartType4/Mass"]) * 10 ** 10

            if gas_ms.size == 0 or star_ms.size == 0 or dm_ms.size == 0:
                continue

            try:
                bh_pos = np.array(bh_data[key]["PartType5/Coordinates"])
                bh_ms = np.array(bh_data[key]["PartType5/Mass"]) * 10 ** 10
            except KeyError:
                bh_pos = np.array([])
                bh_ms = np.array([])

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
            gas_tree = cKDTree(gas_pos)

            # Calculate radii
            dm_rs = np.linalg.norm(dm_pos - cop, axis=1)
            star_rs = np.linalg.norm(all_star_pos - cop, axis=1)
            gas_rs = np.linalg.norm(gas_pos - cop, axis=1)
            if bh_ms.size > 0:
                bh_rs = np.linalg.norm(bh_pos - cop, axis=1)
            else:
                bh_rs = np.array([])

            for (ind, m), fth in zip(enumerate(ini_ms), fths):

                if fth == 0 or m == 0 or gas_ms.size == 0:
                    continue

                # Calculate the number of gas neighbours
                k = int(np.ceil(1.3 * fth))

                # Find neighbouring gas particles to this stellar particle
                _, inds = gas_tree.query(star_pos[ind, :],
                                         k=k,
                                         workers=28)

                # How many neighbours are there?
                if k == 1:

                    i = inds

                    # Get this radius
                    r = gas_rs[i]

                    # Calculate binding energy
                    gas_e_bind = grav_enclosed(r, soft, gas_ms[gas_rs < r],
                                               gas_ms[i], G)
                    dm_e_bind = grav_enclosed(r, soft, dm_ms[dm_rs < r],
                                              gas_ms[i], G)
                    star_e_bind = grav_enclosed(r, soft, star_ms[star_rs < r],
                                                gas_ms[i], G)
                    e_bind = star_e_bind + gas_e_bind + dm_e_bind
                    if bh_ms[bh_rs < r].size > 0:
                        bh_e_bind = grav_enclosed(r, soft, bh_ms[bh_rs < r],
                                                  gas_ms[i], G)
                        e_bind += bh_e_bind

                    binding_energy.append(e_bind)
                    feedback_energy.append(10 ** 51)
                    masses.append(star_data[key]["Mass"] * 10 ** 10)

                else:
                    # Loop over the neigbouring gas particles
                    for i in inds:

                        # Get this radius
                        r = gas_rs[i]

                        # Calculate binding energy
                        gas_e_bind = grav_enclosed(r, soft, gas_ms[gas_rs < r],
                                                   gas_ms[i], G)
                        dm_e_bind = grav_enclosed(r, soft, dm_ms[dm_rs < r],
                                                  gas_ms[i], G)
                        star_e_bind = grav_enclosed(r, soft, star_ms[star_rs < r],
                                                    gas_ms[i], G)
                        e_bind = star_e_bind + gas_e_bind + dm_e_bind
                        if bh_ms.size > 0:
                            if bh_ms[bh_rs < r].size > 0:
                                bh_e_bind = grav_enclosed(r, soft,
                                                          bh_ms[bh_rs < r],
                                                          gas_ms[i], G)
                                e_bind += bh_e_bind

                        binding_energy.append(e_bind)
                        feedback_energy.append(10 ** 51)
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
    ax.set_ylabel(r"$E_{\mathrm{grav}}(<R_\mathrm{gas}) / E_{\mathrm{FB}}$")
    ax.set_xlabel(r"$M_{\star} / M_\odot$")

    ax.legend(loc='upper center',
              bbox_to_anchor=(0.5, -0.2),
              fancybox=True, ncol=3)

    # Save figure
    mkdir("plots/physics_vary/")
    fig.savefig("plots/physics_vary/potential_energy_%s.png" % snap,
                bbox_inches="tight")


def plot_birth_met_vary(stellar_data, snap, path):

    # Define the path
    ini_path = "/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/<type>/data/"

    # Define physics variations directories
    types = ["flares_00", "FLARES_00_REF",
             "FLARES_00_instantFB", "FLARES_00_noZSFthresh",
             "FLARES_00_slightFBlim", "FLARES_00_medFBlim",
             "FLARES_00_highFBlim"]

    # Define labels for each
    labels = ["AGNdT9", "REF", "SKIP",
              "InstantFB", "$Z^0$", "SKIP",
              "$f_{\mathrm{th, max}}=4$", "$f_{\mathrm{th, max}}=6$",
              "$f_{\mathrm{th, max}}=10$"]
    # Define plot dimensions
    nrows = 3
    ncols = 3

    # Define norm
    norm = LogNorm(vmin=1, vmax=50000)

    # Define hexbin extent
    extent = [4.6, 22, 0, 0.119]

    # Set up the plot
    fig = plt.figure(figsize=(nrows * 3.5, ncols * 3.5))
    gs = gridspec.GridSpec(nrows=nrows, ncols=ncols + 1,
                           width_ratios=[20, ] * ncols + [1, ])
    gs.update(wspace=0.0, hspace=0.0)
    axes = []
    cax = fig.add_subplot(gs[-1, -1])

    for i in range(nrows):
        for j in range(ncols):

            if i * ncols + j >= len(labels):
                continue

            if labels[i * ncols + j] == "SKIP":
                continue

            # Create axis
            ax = fig.add_subplot(gs[i, j])

            # Include labels
            if j == 0:
                ax.set_ylabel(r"$Z_{\mathrm{birth}}$")
            if i == nrows - 1:
                ax.set_xlabel(r"$z_{\mathrm{birth}}$")

            # Remove unnecessary ticks
            if j > 0:
                ax.tick_params("y", left=False, right=False,
                               labelleft=False, labelright=False)
            if i < nrows - 1:
                ax.tick_params("x", top=False, bottom=False,
                               labeltop=False, labelbottom=False)

            # Set axis limits
            ax.set_ylim(extent[2], extent[3])
            ax.set_xlim(extent[0], extent[1])

            # Label axis
            ax.text(0.95, 0.9, labels[i * ncols + j],
                    bbox=dict(boxstyle="round,pad=0.3", fc='w', ec="k", lw=1,
                              alpha=0.8),
                    transform=ax.transAxes, horizontalalignment='right',
                    fontsize=8)

            axes.append(ax)

    for (ind, t), l in zip(enumerate(types), labels):

        path = ini_path.replace("<type>", t)

        print(path)

        reg_zs, reg_mets = get_nonmaster_evo_data(
            path, snap, y_key="PartType4/SmoothedMetallicity")

        im = axes[ind].hexbin(reg_zs, reg_mets, mincnt=1, gridsize=50,
                              linewidth=0.2, cmap="plasma",
                              norm=norm, extent=extent)

    # Set up colorbar
    cbar = fig.colorbar(im, cax)
    cbar.set_label("$N$")

    # Save figure
    mkdir("plots/physics_vary/")
    fig.savefig("plots/physics_vary/stellar_birthZ_%s.png" % snap,
                bbox_inches="tight")


def plot_birth_den_vary(stellar_data, snap, path):

    # Define the path
    ini_path = "/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/<type>/data/"

    # Define physics variations directories
    types = ["flares_00", "FLARES_00_REF",
             "FLARES_00_instantFB", "FLARES_00_noZSFthresh",
             "FLARES_00_slightFBlim", "FLARES_00_medFBlim",
             "FLARES_00_highFBlim"]

    # Define labels for each
    labels = ["AGNdT9", "REF", "SKIP",
              "InstantFB", "$Z^0$", "SKIP",
              "$f_{\mathrm{th, max}}=4$", "$f_{\mathrm{th, max}}=6$",
              "$f_{\mathrm{th, max}}=10$"]

    # Define plot dimensions
    nrows = 3
    ncols = 3

    # Define norm
    norm = LogNorm(vmin=1, vmax=50000)

    # Define hexbin extent
    extent = [4.6, 22, -2.2, 5.5]

    # Set up the plot
    fig = plt.figure(figsize=(nrows * 3.5, ncols * 3.5))
    gs = gridspec.GridSpec(nrows=nrows, ncols=ncols + 1,
                           width_ratios=[20, ] * ncols + [1, ])
    gs.update(wspace=0.0, hspace=0.0)
    axes = []
    cax = fig.add_subplot(gs[-1, -1])

    for i in range(nrows):
        for j in range(ncols):

            if i * ncols + j >= len(labels):
                continue

            if labels[i * ncols + j] == "SKIP":
                continue

            # Create axis
            ax = fig.add_subplot(gs[i, j])
            ax.semilogy()

            # Include labels
            if j == 0:
                ax.set_ylabel(r"$n_{\mathrm{H}} / \mathrm{cm}^{-3}$")
            if i == nrows - 1:
                ax.set_xlabel(r"$z_{\mathrm{birth}}$")

            # Remove unnecessary ticks
            if j > 0:
                ax.tick_params("y", left=False, right=False,
                               labelleft=False, labelright=False)
            if i < nrows - 1:
                ax.tick_params("x", top=False, bottom=False,
                               labeltop=False, labelbottom=False)

            # Set axis limits
            ax.set_ylim(10**extent[2], 10**extent[3])
            ax.set_xlim(extent[0], extent[1])

            # Label axis
            ax.text(0.95, 0.9, labels[i * ncols + j],
                    bbox=dict(boxstyle="round,pad=0.3", fc='w', ec="k", lw=1,
                              alpha=0.8),
                    transform=ax.transAxes, horizontalalignment='right',
                    fontsize=8)

            axes.append(ax)

    for (ind, t), l in zip(enumerate(types), labels):

        path = ini_path.replace("<type>", t)

        print(path)

        reg_zs, reg_dens = get_nonmaster_evo_data(
            path, snap, y_key="PartType4/BirthDensity")

        # Convert density to hydrogen number density
        reg_dens = (reg_dens * 10**10
                    * Msun / Mpc ** 3 / mh).to(1 / cm ** 3).value

        im = axes[ind].hexbin(reg_zs, reg_dens, mincnt=1, gridsize=50,
                              yscale="log", linewidth=0.2, cmap="plasma",
                              norm=norm, extent=extent)

    # Set up colorbar
    cbar = fig.colorbar(im, cax)
    cbar.set_label("$N$")

    # Save figure
    mkdir("plots/physics_vary/")
    fig.savefig("plots/physics_vary/stellar_birthden_%s.png" % snap,
                bbox_inches="tight")

    return stellar_data


def plot_ssfr_mass_vary(snap):

    # Define redshift
    z = float(snap.split("z")[-1].replace("p", "."))

    # What redshift was 100 Myrs ago?
    z_100 = z_at_value(cosmo.age, cosmo.age(z) - (0.1 * u.Gyr),
                       zmin=0, zmax=50)

    # Define the path
    ini_path = "/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/<type>/data/"

    # Define physics variations directories
    types = ["flares_00", "FLARES_00_REF",
             "FLARES_00_instantFB", "FLARES_00_noZSFthresh",
             "FLARES_00_slightFBlim", "FLARES_00_medFBlim",
             "FLARES_00_highFBlim"]

    # Define labels for each
    labels = ["AGNdT9", "REF", "SKIP",
              "InstantFB", "$Z^0$", "SKIP",
              "$f_{\mathrm{th, max}}=4$",
              "$f_{\mathrm{th, max}}=6$",
              "$f_{\mathrm{th, max}}=10$"]

    # Define plot dimensions
    nrows = 3
    ncols = 3

    # Define norm
    norm = LogNorm(vmin=1, vmax=10)

    # Define hexbin extent
    extent = [8, 11.5, 0, 15]
    extent1 = [-1.5, 1.5, 0, 15]

    # Set up the plots
    fig = plt.figure(figsize=(nrows * 3.5, ncols * 3.5))
    fig1 = plt.figure(figsize=(nrows * 3.5, ncols * 3.5))
    gs = gridspec.GridSpec(nrows=nrows, ncols=ncols + 1,
                           width_ratios=[20, ] * ncols + [1, ])
    gs.update(wspace=0.0, hspace=0.0)
    gs1 = gridspec.GridSpec(nrows=nrows, ncols=ncols + 1,
                            width_ratios=[20, ] * ncols + [1, ])
    gs1.update(wspace=0.0, hspace=0.0)
    axes = []
    cax = fig.add_subplot(gs[-1, -1])
    axes1 = []
    cax1 = fig1.add_subplot(gs1[-1, -1])

    for i in range(nrows):
        for j in range(ncols):

            if i * ncols + j >= len(labels):
                continue

            if labels[i * ncols + j] == "SKIP":
                continue

            # Create axis
            ax = fig.add_subplot(gs[i, j])
            ax1 = fig1.add_subplot(gs1[i, j])

            # Include labels
            if j == 0:
                ax.set_ylabel(r"$\mathrm{sSFR} / [\mathrm{Gyr}^{-1}]$")
                ax1.set_ylabel(r"$\mathrm{sSFR} / [\mathrm{Gyr}^{-1}]$")
            if i == nrows - 1:
                ax.set_xlabel(r"$M_\star / M_\odot$")
                ax1.set_xlabel(r"$R_{1/2} / [\mathrm{pkpc}]$")

            # Remove unnecessary ticks
            if j > 0:
                ax.tick_params("y", left=False, right=False,
                               labelleft=False, labelright=False)
                ax1.tick_params("y", left=False, right=False,
                                labelleft=False, labelright=False)
            if i < nrows - 1:
                ax.tick_params("x", top=False, bottom=False,
                               labeltop=False, labelbottom=False)
                ax1.tick_params("x", top=False, bottom=False,
                                labeltop=False, labelbottom=False)

            # Set axis limits
            ax.set_ylim(extent[2], extent[3])
            ax.set_xlim(10**extent[0], 10**extent[1])
            ax1.set_ylim(extent1[2], extent1[3])
            ax1.set_xlim(10**extent1[0], 10**extent1[1])

            # Label axis
            ax.text(0.95, 0.9, labels[i * ncols + j],
                    bbox=dict(boxstyle="round,pad=0.3", fc='w', ec="k", lw=1,
                              alpha=0.8),
                    transform=ax.transAxes, horizontalalignment='right',
                    fontsize=8)
            ax1.text(0.95, 0.9, labels[i * ncols + j],
                     bbox=dict(boxstyle="round,pad=0.3", fc='w', ec="k", lw=1,
                               alpha=0.8),
                     transform=ax1.transAxes, horizontalalignment='right',
                     fontsize=8)

            axes.append(ax)
            axes1.append(ax1)

    for (ind, t), l in zip(enumerate(types), labels):

        path = ini_path.replace("<type>", t)

        print(path)

        mass = eagle_io.read_array("SUBFIND", path.replace("<type>", t),
                                   snap,
                                   "Subhalo/ApertureMeasurements/Mass/030kpc",
                                   noH=True, physicalUnits=True,
                                   numThreads=8)[:, 4] * 10 ** 10
        hmrs = eagle_io.read_array("SUBFIND", path.replace("<type>", t),
                                   snap,
                                   "Subhalo/HalfMassRad",
                                   noH=True, physicalUnits=True,
                                   numThreads=8)[:, 4] * 10 ** 3
        cops = eagle_io.read_array("SUBFIND", path.replace("<type>", t),
                                   snap,
                                   "Subhalo/CentreOfPotential",
                                   noH=True, physicalUnits=True,
                                   numThreads=8) * 1000
        grps = eagle_io.read_array("SUBFIND", path.replace("<type>", t),
                                   snap,
                                   "Subhalo/GroupNumber",
                                   noH=True, physicalUnits=True,
                                   numThreads=8)
        subgrps = eagle_io.read_array("SUBFIND", path.replace("<type>", t),
                                      snap,
                                      "Subhalo/SubGroupNumber",
                                      noH=True, physicalUnits=True,
                                      numThreads=8)
        birth_a = eagle_io.read_array("PARTDATA", path.replace("<type>", t),
                                      snap,
                                      "PartType4/StellarFormationTime",
                                      noH=True, physicalUnits=True,
                                      numThreads=8)
        ini_mass = eagle_io.read_array("PARTDATA", path.replace("<type>", t),
                                       snap,
                                       "PartType4/InitialMass",
                                       noH=True, physicalUnits=True,
                                       numThreads=8) * 10 ** 10
        coords = eagle_io.read_array("PARTDATA", path.replace("<type>", t),
                                     snap,
                                     "PartType4/Coordinates",
                                     noH=True, physicalUnits=True,
                                     numThreads=8) * 1000
        part_grps = eagle_io.read_array("PARTDATA", path.replace("<type>", t),
                                        snap,
                                        "PartType4/GroupNumber",
                                        noH=True, physicalUnits=True,
                                        numThreads=8)
        part_subgrps = eagle_io.read_array("PARTDATA",
                                           path.replace("<type>", t),
                                           snap,
                                           "PartType4/SubGroupNumber",
                                           noH=True, physicalUnits=True,
                                           numThreads=8)

        # Apply some cuts
        mokinds = mass > 10**8.5
        mass = mass[mokinds]
        cops = cops[mokinds, :]
        grps = grps[mokinds]
        subgrps = subgrps[mokinds]
        hmrs = hmrs[mokinds]

        # Compute the birth redshift
        birth_z = (1 / birth_a) - 1

        # Get only particles born since z_100
        okinds = birth_z < z_100
        birth_z = birth_z[okinds]
        ini_mass = ini_mass[okinds]
        coords = coords[okinds, :]
        part_grps = part_grps[okinds]
        part_subgrps = part_subgrps[okinds]

        # Set up array to store sfrs
        ssfrs = []
        ms = []
        plt_hmrs = []

        # Loop over galaxies
        for igal in range(mass.size):

            # Get galaxy data
            m = mass[igal]
            cop = cops[igal, :]
            g = grps[igal]
            sg = subgrps[igal]
            hmr = hmrs[igal]

            # Get this galaxies stars
            sokinds = np.logical_and(part_grps == g, part_subgrps == sg)
            this_coords = coords[sokinds, :] - cop
            this_ini_mass = ini_mass[sokinds]

            # Compute stellar radii
            rs = np.sqrt(this_coords[:, 0] ** 2
                         + this_coords[:, 1] ** 2
                         + this_coords[:, 2] ** 2)

            # Get only particles within the aperture
            rokinds = rs < 30
            this_ini_mass = this_ini_mass[rokinds]

            # Compute and store ssfr
            ssfrs.append(np.sum(this_ini_mass) / 0.1 / m)
            ms.append(m)
            plt_hmrs.append(hmr)

        im = axes[ind].hexbin(ms, ssfrs, mincnt=1, gridsize=50,
                              xscale="log", linewidth=0.2,
                              cmap="plasma", norm=norm, extent=extent)
        im1 = axes1[ind].hexbin(plt_hmrs, ssfrs, mincnt=1, gridsize=50,
                                xscale="log", linewidth=0.2,
                                cmap="plasma", norm=norm, extent=extent1)

    # Set up colorbar
    cbar = fig.colorbar(im, cax)
    cbar.set_label("$N$")
    cbar1 = fig1.colorbar(im1, cax1)
    cbar1.set_label("$N$")

    # Save figure
    mkdir("plots/physics_vary/")
    fig.savefig("plots/physics_vary/sfr_mass_%s.png" % snap,
                bbox_inches="tight")
    fig1.savefig("plots/physics_vary/sfr_hmr_%s.png" % snap,
                 bbox_inches="tight")
