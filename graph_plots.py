import numpy as np
import h5py
import matplotlib as mpl
import matplotlib.colors as cm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from flare import plt as flareplt
from utils import mkdir, plot_meidan_stat, calc_ages, plot_spread_stat_as_eb
from astropy.cosmology import Planck18 as cosmo
from astropy.cosmology import z_at_value
import astropy.units as u
import eagle_IO.eagle_IO as eagle_io


def plot_size_change(stellar_data, snaps, plt_type, weight_norm):

    # Define paths
    path = "/cosma/home/dp004/dc-rope1/cosma7/FLARES/flares-mergergraph/"
    halo_base = path + "data/halos/MEGAFLARES_halos_<reg>_<snap>.hdf5"
    graph_base = path + "data/dgraph/MEGAFLARES_graph_<reg>_<snap>.hdf5"

    # Split snapshots into current and progenitor lists
    current_snaps = snaps[1:]
    prog_snaps = snaps[:-1]

    # Initialise lists for storing results
    tot_hmrs = []
    tot_prog_hmrs = []
    tot_cont = []
    tot_mass = []
    w = []

    plt_nprog = []
    plt_hmr = []

    # Loop over snapshots
    for snap, prog_snap in zip(current_snaps, prog_snaps):

        print(snap, prog_snap)

        # Open region 0 initially
        reg = "00"
        print(reg)
        reg_int = 0
        this_halo_base = halo_base.replace("<reg>", reg)
        this_halo_base = this_halo_base.replace("<snap>", snap)
        this_graph_base = graph_base.replace("<reg>", reg)
        this_graph_base = this_graph_base.replace("<snap>", snap)
        this_prog_base = halo_base.replace("<reg>", reg)
        this_prog_base = this_prog_base.replace("<snap>", prog_snap)
        hdf_halo = h5py.File(this_halo_base, "r")
        hdf_prog = h5py.File(this_prog_base, "r")
        hdf_graph = h5py.File(this_graph_base, "r")

        # Get the MEGA ID arrays for both snapshots
        mega_grps = hdf_halo["group_number"][...]
        mega_subgrps = hdf_halo["subgroup_number"][...]
        masses = hdf_halo["masses"][...]
        mega_prog_grps = hdf_prog["group_number"][...]
        mega_prog_subgrps = hdf_prog["subgroup_number"][...]

        # Get the progenitor information
        prog_mass_conts = hdf_graph["ProgMassContribution"][...]
        prog_ids = hdf_graph["ProgHaloIDs"][...]
        start_index = hdf_graph["prog_start_index"][...]
        pmasses = hdf_graph["halo_mass"][...]
        nprogs = hdf_graph["n_progs"][...]

        hdf_halo.close()
        hdf_prog.close()
        hdf_graph.close()

        # Extract galaxy data from the sizes dict
        hmrs = stellar_data[snap]["HMRs"]
        print("There are %d galaxies" % len(hmrs))
        print("There are %d compact galaxies" % len(hmrs[hmrs < 1]))
        prog_hmrs = stellar_data[prog_snap]["HMRs"]
        grps = stellar_data[snap]["Galaxy,GroupNumber"]
        subgrps = stellar_data[snap]["Galaxy,SubGroupNumber"]
        prog_grps = stellar_data[prog_snap]["Galaxy,GroupNumber"]
        prog_subgrps = stellar_data[prog_snap]["Galaxy,SubGroupNumber"]
        regions = stellar_data[snap]["regions"]
        ws = stellar_data[snap]["weights"]
        prog_regions = stellar_data[prog_snap]["regions"]

        # Loop over galaxies
        for ind in range(len(hmrs)):

            # Skip this galaxy if it is not compact
            if hmrs[ind] > 1:
                continue

            # Get the region for this galaxy
            reg_int = regions[ind]
            if reg_int == 18:
                continue
            if int(reg) != reg_int:
                reg = str(reg_int).zfill(2)
                print(reg)

                if reg_int == 18:
                    continue

                # Open this new region
                this_halo_base = halo_base.replace("<reg>", reg)
                this_halo_base = this_halo_base.replace("<snap>", snap)
                this_graph_base = graph_base.replace("<reg>", reg)
                this_graph_base = this_graph_base.replace("<snap>", snap)
                this_prog_base = halo_base.replace("<reg>", reg)
                this_prog_base = this_prog_base.replace("<snap>", prog_snap)
                hdf_halo = h5py.File(this_halo_base, "r")
                hdf_prog = h5py.File(this_prog_base, "r")
                hdf_graph = h5py.File(this_graph_base, "r")

                # Get the MEGA ID arrays for both snapshots
                mega_grps = hdf_halo["group_number"][...]
                mega_subgrps = hdf_halo["subgroup_number"][...]
                masses = hdf_halo["masses"][...]
                mega_prog_grps = hdf_prog["group_number"][...]
                mega_prog_subgrps = hdf_prog["subgroup_number"][...]

                # Get the progenitor information
                prog_mass_conts = hdf_graph["ProgMassContribution"][...]
                prog_ids = hdf_graph["ProgHaloIDs"][...]
                start_index = hdf_graph["prog_start_index"][...]
                pmasses = hdf_graph["halo_mass"][...]
                nprogs = hdf_graph["n_progs"][...]

                hdf_halo.close()
                hdf_prog.close()
                hdf_graph.close()

            # Extract this galaxies information
            hmr = hmrs[ind]
            g, sg = grps[ind], subgrps[ind]

            # Whats the MEGA ID of this galaxy?
            mega_ind = np.where(np.logical_and(mega_grps == g,
                                               mega_subgrps == sg))[0]

            # Get the progenitor
            start = start_index[mega_ind][0]
            stride = nprogs[mega_ind][0]
            main_prog = prog_ids[start]
            star_m = pmasses[mega_ind, 4] * 10 ** 10

            if stride == 0:
                plt_nprog.append(stride)
                plt_hmr.append(hmr)
                continue

            # Get this progenitors group and subgroup ID
            prog_g = mega_prog_grps[main_prog]
            prog_sg = mega_prog_subgrps[main_prog]

            # Get this progenitor's size
            flares_ind = np.where(
                np.logical_and(prog_regions == reg_int,
                               np.logical_and(prog_grps == prog_g,
                                              prog_subgrps == prog_sg))
            )[0]
            prog_hmr = prog_hmrs[flares_ind]

            if prog_hmr.size == 0:
                continue

            # Get the contribution information
            prog_cont = prog_mass_conts[start: start + stride] * 10 ** 10
            mass = masses[mega_ind] * 10 ** 10

            # Calculate the mass contribution as a fraction of current mass
            star_prog_cont = np.sum(prog_cont[:, 4])
            frac_prog_cont = star_prog_cont / star_m

            tot_prog_cont = np.sum(prog_cont, axis=1)
            frac_tot_prog_cont = tot_prog_cont / mass

            plt_nprog.append(tot_prog_cont[frac_tot_prog_cont > 0.1].size)
            plt_hmr.append(hmr)

            # Include these results for plotting
            tot_cont.extend(frac_prog_cont)
            tot_hmrs.append(hmr)
            tot_prog_hmrs.extend(prog_hmr)
            tot_mass.append(star_m)
            w.append(ws[ind])

    # Convert to arrays
    print(len(tot_hmrs))
    print(len(tot_cont))
    print(len(tot_prog_hmrs))
    tot_hmrs = np.array(tot_hmrs)
    tot_prog_hmrs = np.array(tot_prog_hmrs)
    tot_cont = np.array(tot_cont)
    tot_mass = np.array(tot_mass)
    w = np.array(w)

    # Compute delta
    delta_hmr = tot_hmrs - tot_prog_hmrs

    # Set up plot
    fig = plt.figure(figsize=(3.5, 3.5))
    ax = fig.add_subplot(111)

    # Plot the scatter
    im = ax.hexbin(tot_cont, delta_hmr,  gridsize=30,
                   mincnt=np.min(w) - (0.1 * np.min(w)),
                   C=w, extent=[-1, 1.3, -1, 1.3],
                   reduce_C_function=np.sum, xscale="log",
                   linewidths=0.2, norm=weight_norm)

    # Axes labels
    ax.set_xlabel("$M_{A,\star} / M_\mathrm{B,\star}$")
    ax.set_ylabel("$\Delta R_{1/2} / [\mathrm{pkpc}]$")

    cbar = fig.colorbar(im)
    cbar.set_label("$\log_{10}(M_{\star} / M_\odot)$")

    # Save figure
    mkdir("plots/graph/")
    fig.savefig("plots/graph/delta_hmr_contribution_%s.png" % plt_type,
                bbox_inches="tight")
    plt.close(fig)

    # Set up plot
    fig = plt.figure(figsize=(3.5, 3.5))
    ax = fig.add_subplot(111)
    ax.semilogx()

    # Plot the scatter
    ax.scatter(tot_mass, delta_hmr, marker=".")

    # Axes labels
    ax.set_xlabel("$M_{\star} / M_\odot$")
    ax.set_ylabel("$\Delta R_{1/2} / [\mathrm{pkpc}]$")

    # Save figure
    mkdir("plots/graph/")
    fig.savefig("plots/graph/delta_hmr_stellarmass_%s.png" % plt_type,
                bbox_inches="tight")
    plt.close(fig)

    # Set up plot
    fig = plt.figure(figsize=(3.5, 3.5))
    ax = fig.add_subplot(111)

    # Plot the scatter
    ax.scatter(tot_hmrs, delta_hmr, marker=".")

    # Axes labels
    ax.set_xlabel("$R_{1/2} / [\mathrm{pkpc}]$")
    ax.set_ylabel("$\Delta R_{1/2} / [\mathrm{pkpc}]$")

    # Save figure
    mkdir("plots/graph/")
    fig.savefig("plots/graph/delta_hmr_hmrs.png",
                bbox_inches="tight")
    plt.close(fig)

    # Set up plot
    fig = plt.figure(figsize=(3.5, 3.5))
    ax = fig.add_subplot(111)

    # Plot the scatter
    ax.scatter(tot_prog_hmrs, delta_hmr, marker=".")

    # Axes labels
    ax.set_xlabel("$R_{1/2, prog} / [\mathrm{pkpc}]$")
    ax.set_ylabel("$\Delta R_{1/2} / [\mathrm{pkpc}]$")

    # Save figure
    mkdir("plots/graph/")
    fig.savefig("plots/graph/delta_hmr_proghmrs_%s.png" % plt_type,
                bbox_inches="tight")
    plt.close(fig)

    # Set up plot
    print(np.unique(plt_nprog, return_counts=True))
    fig = plt.figure(figsize=(3.5, 3.5))
    ax = fig.add_subplot(111)

    # Plot the scatter
    ax.scatter(plt_nprog, plt_hmr, marker=".")

    # Axes labels
    ax.set_xlabel("$R_{1/2} / [\mathrm{pkpc}]$")
    ax.set_ylabel("$N_{\mathrm{prog}}$")

    # Save figure
    mkdir("plots/graph/")
    fig.savefig("plots/graph/nprog_hmr_%s.png" % plt_type,
                bbox_inches="tight")
    plt.close(fig)


def plot_size_change_comp(stellar_data, gas_data, snaps):

    # Define paths
    path = "/cosma/home/dp004/dc-rope1/cosma7/FLARES/flares-mergergraph/"
    halo_base = path + "data/halos/MEGAFLARES_halos_<reg>_<snap>.hdf5"
    graph_base = path + "data/dgraph/MEGAFLARES_graph_<reg>_<snap>.hdf5"

    # Split snapshots into current and progenitor lists
    current_snaps = snaps[1:]
    prog_snaps = snaps[:-1]

    # Initialise lists for storing results
    tot_hmrs = {"star": [], "gas": []}
    tot_prog_hmrs = {"star": [], "gas": []}
    tot_cont = []

    # Loop over snapshots
    for snap, prog_snap in zip(current_snaps, prog_snaps):

        print(snap, prog_snap)

        # Open region 0 initially
        reg = "00"
        print(reg)
        reg_int = 0
        this_halo_base = halo_base.replace("<reg>", reg)
        this_halo_base = this_halo_base.replace("<snap>", snap)
        this_graph_base = graph_base.replace("<reg>", reg)
        this_graph_base = this_graph_base.replace("<snap>", snap)
        this_prog_base = halo_base.replace("<reg>", reg)
        this_prog_base = this_prog_base.replace("<snap>", prog_snap)
        hdf_halo = h5py.File(this_halo_base, "r")
        hdf_prog = h5py.File(this_prog_base, "r")
        hdf_graph = h5py.File(this_graph_base, "r")

        # Get the MEGA ID arrays for both snapshots
        mega_grps = hdf_halo["group_number"][...]
        mega_subgrps = hdf_halo["subgroup_number"][...]
        masses = hdf_halo["masses"][...]
        mega_prog_grps = hdf_prog["group_number"][...]
        mega_prog_subgrps = hdf_prog["subgroup_number"][...]

        # Get the progenitor information
        prog_mass_conts = hdf_graph["ProgMassContribution"][...]
        prog_ids = hdf_graph["ProgHaloIDs"][...]
        start_index = hdf_graph["prog_start_index"][...]
        pmasses = hdf_graph["halo_mass"][...]
        nprogs = hdf_graph["n_progs"][...]

        hdf_halo.close()
        hdf_prog.close()
        hdf_graph.close()

        # Extract galaxy data from the sizes dict
        gas_hmrs = gas_data[snap]["HMRs"]
        star_hmrs = stellar_data[snap]["HMRs"]
        print("There are %d galaxies (%d)" % (len(star_hmrs), len(gas_hmrs)))
        print("There are %d compact galaxies" % len(star_hmrs[star_hmrs < 1]))
        star_prog_hmrs = stellar_data[prog_snap]["HMRs"]
        star_grps = stellar_data[snap]["Galaxy,GroupNumber"]
        star_subgrps = stellar_data[snap]["Galaxy,SubGroupNumber"]
        star_prog_grps = stellar_data[prog_snap]["Galaxy,GroupNumber"]
        star_prog_subgrps = stellar_data[prog_snap]["Galaxy,SubGroupNumber"]
        star_regions = stellar_data[snap]["regions"]
        star_prog_regions = stellar_data[prog_snap]["regions"]
        gas_prog_hmrs = gas_data[prog_snap]["HMRs"]
        gas_grps = gas_data[snap]["Galaxy,GroupNumber"]
        gas_subgrps = gas_data[snap]["Galaxy,SubGroupNumber"]
        gas_prog_grps = gas_data[prog_snap]["Galaxy,GroupNumber"]
        gas_prog_subgrps = gas_data[prog_snap]["Galaxy,SubGroupNumber"]
        gas_regions = gas_data[snap]["regions"]
        gas_prog_regions = gas_data[prog_snap]["regions"]

        # Loop over galaxies
        for ind in range(len(star_hmrs)):

            # Skip this galaxy if it is not compact
            if star_hmrs[ind] > 1:
                continue

            # Get the region for this galaxy
            reg_int = star_regions[ind]
            if reg_int == 18:
                continue
            if int(reg) != reg_int:
                reg = str(reg_int).zfill(2)
                print(reg)

                if reg_int == 18:
                    continue

                # Open this new region
                this_halo_base = halo_base.replace("<reg>", reg)
                this_halo_base = this_halo_base.replace("<snap>", snap)
                this_graph_base = graph_base.replace("<reg>", reg)
                this_graph_base = this_graph_base.replace("<snap>", snap)
                this_prog_base = halo_base.replace("<reg>", reg)
                this_prog_base = this_prog_base.replace(
                    "<snap>", prog_snap)
                hdf_halo = h5py.File(this_halo_base, "r")
                hdf_prog = h5py.File(this_prog_base, "r")
                hdf_graph = h5py.File(this_graph_base, "r")

                # Get the MEGA ID arrays for both snapshots
                mega_grps = hdf_halo["group_number"][...]
                mega_subgrps = hdf_halo["subgroup_number"][...]
                masses = hdf_halo["masses"][...]
                mega_prog_grps = hdf_prog["group_number"][...]
                mega_prog_subgrps = hdf_prog["subgroup_number"][...]

                # Get the progenitor information
                prog_mass_conts = hdf_graph["ProgMassContribution"][...]
                prog_ids = hdf_graph["ProgHaloIDs"][...]
                start_index = hdf_graph["prog_start_index"][...]
                pmasses = hdf_graph["halo_mass"][...]
                nprogs = hdf_graph["n_progs"][...]

                hdf_halo.close()
                hdf_prog.close()
                hdf_graph.close()

            # Extract this galaxies information
            star_hmr = star_hmrs[ind]
            gas_hmr = gas_hmrs[ind]
            g, sg = star_grps[ind], star_subgrps[ind]

            # Whats the MEGA ID of this galaxy?
            mega_ind = np.where(np.logical_and(mega_grps == g,
                                               mega_subgrps == sg))[0]

            # Get the progenitor
            start = start_index[mega_ind][0]
            stride = nprogs[mega_ind][0]
            main_prog = prog_ids[start]
            star_m = pmasses[mega_ind, 4] * 10 ** 10

            # Get this progenitors group and subgroup ID
            prog_g = mega_prog_grps[main_prog]
            prog_sg = mega_prog_subgrps[main_prog]

            # Get this progenitor's size
            sflares_ind = np.where(
                np.logical_and(star_prog_regions == reg_int,
                               np.logical_and(star_prog_grps == prog_g,
                                              star_prog_subgrps == prog_sg))
            )[0]
            gflares_ind = np.where(
                np.logical_and(gas_prog_regions == reg_int,
                               np.logical_and(gas_prog_grps == prog_g,
                                              gas_prog_subgrps == prog_sg))
            )[0]
            star_prog_hmr = star_prog_hmrs[sflares_ind]
            gas_prog_hmr = gas_prog_hmrs[gflares_ind]

            if star_prog_hmr.size == 0:
                continue

            # Get the contribution information
            prog_cont = prog_mass_conts[start: start + stride] * 10 ** 10
            mass = masses[mega_ind] * 10 ** 10

            # Calculate the mass contribution as a fraction of current mass
            star_prog_cont = np.sum(prog_cont[:, 4])
            frac_prog_cont = star_prog_cont / mass

            # Include these results for plotting
            tot_cont.extend(frac_prog_cont)
            tot_hmrs["star"].append(star_hmr)
            tot_prog_hmrs["star"].extend(star_prog_hmr)
            tot_hmrs["gas"].append(gas_hmr)
            tot_prog_hmrs["gas"].extend(gas_prog_hmr)

    # Convert to arrays
    gas_tot_hmrs = np.array(tot_hmrs["gas"])
    gas_tot_prog_hmrs = np.array(tot_prog_hmrs["gas"])
    star_tot_hmrs = np.array(tot_hmrs["star"])
    star_tot_prog_hmrs = np.array(tot_prog_hmrs["star"])
    tot_cont = np.array(tot_cont)

    # Get deltas
    gas_delta_hmr = gas_tot_hmrs - gas_tot_prog_hmrs
    star_delta_hmr = star_tot_hmrs - star_tot_prog_hmrs

    # Set up plot
    fig = plt.figure(figsize=(3.5, 3.5))
    ax = fig.add_subplot(111)

    # Plot the scatter
    im = ax.hexbin(gas_delta_hmr, star_delta_hmr, gridsize=50,
                   mincnt=np.min(tot_cont) - (0.1 * np.min(tot_cont)),
                   C=tot_cont, norm=cm.LogNorm(),
                   reduce_C_function=np.mean,
                   linewidths=0.2)

    # Axes labels
    ax.set_xlabel("$\Delta R_\mathrm{gas} / [\mathrm{pkpc}]$")
    ax.set_ylabel("$\Delta R_\star / [\mathrm{pkpc}]$")

    cbar = fig.colorbar(im)
    cbar.set_label("$\sum_p M_\mathrm{\star}^{p} / M_\star$")

    # Save figure
    mkdir("plots/graph/")
    fig.savefig("plots/graph/delta_hmr_comp.png",
                bbox_inches="tight")
    plt.close(fig)
