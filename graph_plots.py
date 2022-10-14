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
import astropy.constants as const
import eagle_IO.eagle_IO as eagle_io
from utils import grav
import cmasher as cmr
from scipy.spatial.distance import cdist
from mpi4py import MPI


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

    # Physical constants
    G = (const.G.to(u.Mpc ** 3 * u.M_sun ** -1 * u.yr ** -2))

    # Loop over snapshots
    for snap, prog_snap in zip(current_snaps, prog_snaps):

        print(snap, prog_snap)

        if snap != current_snaps[-1]:
            continue

        # Get redshift
        z = float(snap.split("z")[-1].replace("p", "."))

        # Define softening length
        if z <= 2.8:
            soft = 0.000474390 / 0.6777
        else:
            soft = 0.001802390 / (0.6777 * (1 + z))

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
        hmrs = stellar_data[snap]["HMRs"][...]
        print("There are %d galaxies" % len(hmrs))
        print("There are %d compact galaxies" % len(hmrs[hmrs < 1]))
        prog_hmrs = stellar_data[prog_snap]["HMRs"][...]
        grps = stellar_data[snap]["Galaxy,GroupNumber"][...]
        subgrps = stellar_data[snap]["Galaxy,SubGroupNumber"][...]
        prog_grps = stellar_data[prog_snap]["Galaxy,GroupNumber"][...]
        prog_subgrps = stellar_data[prog_snap]["Galaxy,SubGroupNumber"][...]
        regions = stellar_data[snap]["regions"][...]
        ws = stellar_data[snap]["weights"][...]
        prog_regions = stellar_data[prog_snap]["regions"][...]

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
                   C=w,
                   reduce_C_function=np.sum,
                   linewidths=0.2, norm=weight_norm, cmap="plasma")

    # Axes labels
    ax.set_xlabel("$M_{A,\star} / M_\mathrm{B,\star}$")
    ax.set_ylabel("$\Delta R_{1/2} / [\mathrm{pkpc}]$")

    cbar = fig.colorbar(im)
    cbar.set_label("$\sum w_{i}$")

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


def plot_size_change_comp(stellar_data, gas_data, snaps, weight_norm):

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
    tot_gas_mass = []
    tot_prog_gas_mass = []
    w = []

    # Loop over snapshots
    for snap, prog_snap in zip(current_snaps, prog_snaps):

        print(snap, prog_snap)

        if snap != current_snaps[-1]:
            continue

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
        gas_hmrs = gas_data[snap]["HMRs"][...]
        star_hmrs = stellar_data[snap]["HMRs"][...]
        print("There are %d galaxies (%d)" % (len(star_hmrs), len(gas_hmrs)))
        print("There are %d compact galaxies" % len(star_hmrs[star_hmrs < 1]))
        star_prog_hmrs = stellar_data[prog_snap]["HMRs"][...]
        star_grps = stellar_data[snap]["Galaxy,GroupNumber"][...]
        star_subgrps = stellar_data[snap]["Galaxy,SubGroupNumber"][...]
        star_prog_grps = stellar_data[prog_snap]["Galaxy,GroupNumber"][...]
        star_prog_subgrps = stellar_data[prog_snap]["Galaxy,SubGroupNumber"][...]
        star_regions = stellar_data[snap]["regions"][...]
        star_prog_regions = stellar_data[prog_snap]["regions"][...]
        gas_prog_hmrs = gas_data[prog_snap]["HMRs"][...]
        gas_grps = gas_data[snap]["Galaxy,GroupNumber"][...]
        gas_subgrps = gas_data[snap]["Galaxy,SubGroupNumber"][...]
        gas_prog_grps = gas_data[prog_snap]["Galaxy,GroupNumber"][...]
        gas_prog_subgrps = gas_data[prog_snap]["Galaxy,SubGroupNumber"][...]
        gas_regions = gas_data[snap]["regions"][...]
        gas_prog_regions = gas_data[prog_snap]["regions"][...]
        ws = stellar_data[snap]["weights"][...]

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
            tot_gas_mass
            w.append(ws[ind])

    # Convert to arrays
    gas_tot_hmrs = np.array(tot_hmrs["gas"])
    gas_tot_prog_hmrs = np.array(tot_prog_hmrs["gas"])
    star_tot_hmrs = np.array(tot_hmrs["star"])
    star_tot_prog_hmrs = np.array(tot_prog_hmrs["star"])
    tot_cont = np.array(tot_cont)
    w = np.array(w)

    # Get deltas
    gas_delta_hmr = gas_tot_hmrs / gas_tot_prog_hmrs
    star_delta_hmr = star_tot_hmrs / star_tot_prog_hmrs

    # Remove zeros
    okinds = np.logical_and(gas_delta_hmr > 0, star_delta_hmr > 0)

    # Set up plot
    fig = plt.figure(figsize=(3.5, 3.5))
    ax = fig.add_subplot(111)

    # Plot the scatter
    im = ax.hexbin(gas_delta_hmr[okinds], star_delta_hmr[okinds], gridsize=30,
                   mincnt=np.min(w) - (0.1 * np.min(w)),
                   C=w[okinds], xscale="log", yscale="log",
                   reduce_C_function=np.sum,
                   linewidths=0.2, norm=weight_norm, cmap="plasma")

    # Axes labels
    ax.set_xlabel(
        "$R_\mathrm{1/2, \mathrm{gas}}^{B} / R_\mathrm{1/2, \mathrm{gas}}^{A}$")
    ax.set_ylabel("$R_\mathrm{1/2, \star}^{B} / R_\mathrm{1/2, \star}^{A}$")

    cbar = fig.colorbar(im)
    cbar.set_label("$\sum w_{i}$")

    # Save figure
    mkdir("plots/graph/")
    fig.savefig("plots/graph/delta_hmr_comp.png",
                bbox_inches="tight")
    plt.close(fig)


def plot_size_change_binding(stellar_data, snaps, weight_norm, comm, nranks, rank):

    # Get the dark matter mass
    hdf = h5py.File("/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/flares_00/"
                    "data/snapshot_000_z015p000/snap_000_z015p000.0.hdf5")
    mdm = hdf["Header"].attrs["MassTable"][1]
    hdf.close()

    # Define paths
    path = "/cosma/home/dp004/dc-rope1/cosma7/FLARES/flares-mergergraph/"
    halo_base = path + "data/halos/MEGAFLARES_halos_<reg>_<snap>.hdf5"
    graph_base = path + "data/dgraph/MEGAFLARES_graph_<reg>_<snap>.hdf5"
    master_base = "/cosma7/data/dp004/dc-payy1/my_files/flares_pipeline/data/flares.hdf5"

    # Split snapshots into current and progenitor lists
    current_snaps = snaps[1:]
    prog_snaps = snaps[:-1]

    # Initialise lists for storing results
    tot_hmrs = []
    tot_prog_hmrs = []
    binding_energy = []
    feedback_energy = []
    prog_binding_energy = []
    prog_feedback_energy = []
    w = []

    # Open the master file
    hdf_master = h5py.File(master_base, "r")

    # Physical constants
    G = (const.G.to(u.Mpc ** 3 * u.M_sun ** -1 * u.Myr ** -2)).value

    # Loop over snapshots
    for snap, prog_snap in zip(current_snaps, prog_snaps):

        if snap != current_snaps[-1]:
            continue

        if rank == 0:
            print(snap, prog_snap)

        # Get redshift
        z = float(snap.split("z")[-1].replace("p", "."))
        prog_z = float(prog_snap.split("z")[-1].replace("p", "."))

        # Define softening length
        soft = 0.001802390 / (0.6777 * (1 + z))
        prog_soft = 0.001802390 / (0.6777 * (1 + prog_z))

        # Set fake region IDs
        reg = "100"
        reg_int = -1

        if snap != snaps[-1]:
            continue

        # Extract galaxy data from the sizes dict
        hmrs = stellar_data[snap]["HMRs"][...]
        if rank == 0:
            print("There are %d galaxies" % len(hmrs))
            print("There are %d compact galaxies" % len(hmrs[hmrs < 1]))
        prog_hmrs = stellar_data[prog_snap]["HMRs"][...]
        grps = stellar_data[snap]["Galaxy,GroupNumber"][...]
        subgrps = stellar_data[snap]["Galaxy,SubGroupNumber"][...]
        prog_grps = stellar_data[prog_snap]["Galaxy,GroupNumber"][...]
        prog_subgrps = stellar_data[prog_snap]["Galaxy,SubGroupNumber"][...]
        regions = stellar_data[snap]["regions"][...]
        ws = stellar_data[snap]["weights"][...]
        prog_regions = stellar_data[prog_snap]["regions"][...]

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
                if rank == 0:
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

                # Get the master file group
                snap_grp = hdf_master[reg][snap]
                prog_grp = hdf_master[reg][prog_snap]

                # Get the group and subgroup id arrays from the master file
                master_grps = snap_grp["Galaxy"]["GroupNumber"][...]
                master_subgrps = snap_grp["Galaxy"]["SubGroupNumber"][...]
                prog_master_grps = prog_grp["Galaxy"]["GroupNumber"][...]
                prog_master_subgrps = prog_grp["Galaxy"]["SubGroupNumber"][...]

                # Get other data from the master file
                cops = snap_grp["Galaxy"]["COP"][...].T / (1 + z)
                master_s_length = snap_grp["Galaxy"]["S_Length"][...]
                master_s_pos = snap_grp["Particle"]["S_Coordinates"][...].T / \
                    (1 + z)
                ini_ms = snap_grp["Particle"]["S_MassInitial"][...] * 10 ** 10
                s_mass = snap_grp["Particle"]["S_Mass"][...] * 10 ** 10
                master_g_length = snap_grp["Galaxy"]["G_Length"][...]
                master_g_pos = snap_grp["Particle"]["G_Coordinates"][...].T / \
                    (1 + z)
                g_mass = snap_grp["Particle"]["G_Mass"][...] * 10 ** 10
                master_dm_length = snap_grp["Galaxy"]["DM_Length"][...]
                master_dm_pos = snap_grp["Particle"]["DM_Coordinates"][...].T / (
                    1 + z)
                dm_mass = np.full(master_dm_pos.shape[0], mdm * 10 ** 10)
                master_bh_length = snap_grp["Galaxy"]["BH_Length"][...]
                master_bh_pos = snap_grp["Particle"]["BH_Coordinates"][...].T / (
                    1 + z)
                bh_mass = snap_grp["Particle"]["BH_Mass"][...] * 10 ** 10
                prog_cops = prog_grp["Galaxy"]["COP"][...].T / (1 + prog_z)
                prog_master_s_length = prog_grp["Galaxy"]["S_Length"][...]
                prog_master_s_pos = prog_grp["Particle"]["S_Coordinates"][...].T / (
                    1 + prog_z)
                prog_ini_ms = prog_grp["Particle"]["S_MassInitial"][...] * 10 ** 10
                prog_s_mass = prog_grp["Particle"]["S_Mass"][...] * 10 ** 10
                prog_master_g_length = prog_grp["Galaxy"]["G_Length"][...]
                prog_master_g_pos = prog_grp["Particle"]["G_Coordinates"][...].T / (
                    1 + prog_z)
                prog_g_mass = prog_grp["Particle"]["G_Mass"][...] * 10 ** 10
                prog_master_dm_length = prog_grp["Galaxy"]["DM_Length"][...]
                prog_master_dm_pos = prog_grp["Particle"]["DM_Coordinates"][...].T / (
                    1 + prog_z)
                prog_dm_mass = np.full(
                    prog_master_dm_pos.shape[0], mdm * 10 ** 10)
                prog_master_bh_length = prog_grp["Galaxy"]["BH_Length"][...]
                prog_master_bh_pos = prog_grp["Particle"]["BH_Coordinates"][...].T / (
                    1 + prog_z)
                prog_bh_mass = prog_grp["Particle"]["BH_Mass"][...] * 10 ** 10

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

            # Get the index in the master file
            master_ind = np.where(
                np.logical_and(master_grps == g,
                               master_subgrps == sg)
            )[0]
            prog_master_ind = np.where(
                np.logical_and(prog_master_grps == prog_g,
                               prog_master_subgrps == prog_sg)
            )[0]

            if master_ind.size == 0:
                continue
            if prog_master_ind.size == 0:
                continue

            # Extract the index from the array it's contained within
            master_ind = master_ind[0]
            prog_master_ind = prog_master_ind[0]

            # Get the start index for each particle type
            s_start = np.sum(master_s_length[:master_ind])
            g_start = np.sum(master_g_length[:master_ind])
            dm_start = np.sum(master_dm_length[:master_ind])
            bh_start = np.sum(master_bh_length[:master_ind])
            s_len = master_s_length[master_ind]
            g_len = master_g_length[master_ind]
            dm_len = master_dm_length[master_ind]
            bh_len = master_bh_length[master_ind]
            cop = cops[master_ind]

            prog_s_start = np.sum(prog_master_s_length[:prog_master_ind])
            prog_g_start = np.sum(prog_master_g_length[:prog_master_ind])
            prog_dm_start = np.sum(prog_master_dm_length[:prog_master_ind])
            prog_bh_start = np.sum(prog_master_bh_length[:prog_master_ind])
            prog_s_len = prog_master_s_length[prog_master_ind]
            prog_g_len = prog_master_g_length[prog_master_ind]
            prog_dm_len = prog_master_dm_length[prog_master_ind]
            prog_bh_len = prog_master_bh_length[prog_master_ind]
            prog_cop = prog_cops[prog_master_ind]

            # Get this galaxy's data
            this_s_pos = master_s_pos[s_start: s_start + s_len, :]
            this_g_pos = master_g_pos[g_start: g_start + g_len, :]
            this_dm_pos = master_dm_pos[dm_start: dm_start + dm_len, :]
            this_bh_pos = master_bh_pos[bh_start: bh_start + bh_len, :]
            this_s_mass = s_mass[s_start: s_start + s_len]
            this_g_mass = g_mass[g_start: g_start + g_len]
            this_dm_mass = dm_mass[dm_start: dm_start + dm_len]
            this_bh_mass = bh_mass[bh_start: bh_start + bh_len]
            this_ini_ms = ini_ms[s_start: s_start + s_len]

            prog_this_s_pos = prog_master_s_pos[
                prog_s_start: prog_s_start + prog_s_len, :]
            prog_this_g_pos = prog_master_g_pos[
                prog_g_start: prog_g_start + prog_g_len, :]
            prog_this_dm_pos = prog_master_dm_pos[
                prog_dm_start: prog_dm_start + prog_dm_len, :]
            prog_this_bh_pos = prog_master_bh_pos[
                prog_bh_start: prog_bh_start + prog_bh_len, :]
            prog_this_s_mass = prog_s_mass[
                prog_s_start: prog_s_start + prog_s_len]
            prog_this_g_mass = prog_g_mass[
                prog_g_start: prog_g_start + prog_g_len]
            prog_this_dm_mass = prog_dm_mass[
                prog_dm_start: prog_dm_start + prog_dm_len]
            prog_this_bh_mass = prog_bh_mass[
                prog_bh_start: prog_bh_start + prog_bh_len]
            prog_this_ini_ms = prog_ini_ms[
                prog_s_start: prog_s_start + prog_s_len]

            # Combine coordiantes and masses into a single array
            part_counts = [g_len,
                           dm_len,
                           0, 0,
                           s_len,
                           bh_len]
            prog_part_counts = [prog_g_len,
                                prog_dm_len,
                                0, 0,
                                prog_s_len,
                                prog_bh_len]
            npart = np.sum(part_counts)
            prog_npart = np.sum(prog_part_counts)
            coords = np.zeros((npart, 3))
            prog_coords = np.zeros((prog_npart, 3))
            masses = np.zeros(npart)
            prog_masses = np.zeros(prog_npart)

            # Create lists of particle type data
            lst_coords = (this_g_pos, this_dm_pos, [], [],
                          this_s_pos, this_bh_pos)
            prog_lst_coords = (prog_this_g_pos, prog_this_dm_pos, [], [],
                               prog_this_s_pos, prog_this_bh_pos)
            lst_masses = (this_g_mass, this_dm_mass, [], [],
                          this_s_mass, this_bh_mass)
            prog_lst_masses = (prog_this_g_mass, prog_this_dm_mass, [], [],
                               prog_this_s_mass, prog_this_bh_mass)

            # Construct combined arrays
            for pos, ms, ipart in zip(lst_coords, lst_masses,
                                      range(len(prog_part_counts))):
                if ipart > 0:
                    low = part_counts[ipart - 1]
                    high = part_counts[ipart - 1] + part_counts[ipart]
                else:
                    low, high = 0, part_counts[ipart]
                if part_counts[ipart] > 0:
                    coords[low:  high, :] = pos
                    masses[low: high] = ms

            # Construct combined prog arrays
            for pos, ms, ipart in zip(prog_lst_coords, prog_lst_masses,
                                      range(len(prog_part_counts))):
                if ipart > 0:
                    low = prog_part_counts[ipart - 1]
                    high = prog_part_counts[ipart -
                                            1] + prog_part_counts[ipart]
                else:
                    low, high = 0, prog_part_counts[ipart]
                if prog_part_counts[ipart] > 0:
                    prog_coords[low: high, :] = pos
                    prog_masses[low: high] = ms

            # Calculate radius and apply a 30 pkpc aperture
            rs = np.sqrt((coords[:, 0] - cop[0]) ** 2
                         + (coords[:, 1] - cop[1]) ** 2
                         + (coords[:, 2] - cop[2]) ** 2)
            prog_rs = np.sqrt((prog_coords[:, 0] - prog_cop[0]) ** 2
                              + (prog_coords[:, 1] - prog_cop[1]) ** 2
                              + (prog_coords[:, 2] - prog_cop[2]) ** 2)
            okinds = rs < 0.03
            prog_okinds = prog_rs < 0.03

            # Get only particles within the aperture
            coords = coords[okinds]
            masses = masses[okinds]
            prog_coords = prog_coords[prog_okinds]
            prog_masses = prog_masses[prog_okinds]

            # Get the particles this rank has to handle
            rank_parts = np.linspace(0, masses.size, nranks + 1, dtype=int)
            prog_rank_parts = np.linspace(
                0, prog_masses.size, nranks + 1, dtype=int)

            # Define gravitational binding energy
            ebind = 0
            prog_ebind = 0

            # Loop over my particles
            for pind in range(rank_parts[rank], rank_parts[rank + 1]):

                # Get this particle
                pos_i = np.array([coords[pind, :], ])
                mass_i = masses[pind]

                # Get distances
                dists = cdist(pos_i, coords, metric="sqeuclidean")

                # Add this contribution to the binding energy
                ebind += np.sum(masses * mass_i
                                / np.sqrt(dists + soft ** 2))

            # Loop over my particles for progenitors
            for pind in range(prog_rank_parts[rank], prog_rank_parts[rank + 1]):

                # Get this particle
                pos_i = np.array([prog_coords[pind, :], ])
                mass_i = prog_masses[pind]

                # Get distances
                dists = cdist(pos_i, prog_coords, metric="sqeuclidean")

                # Add this contribution to the binding energy
                prog_ebind += np.sum(prog_masses * mass_i
                                     / np.sqrt(dists + prog_soft ** 2))

            # Complete the calculation
            ebind *= G / 2 * u.Msun * u.Mpc ** 2 * u.Myr ** -2
            prog_ebind *= G / 2 * u.Msun * u.Mpc ** 2 * u.Myr ** -2

            # Convert binding energy units
            ebind = ebind.to(u.erg).value
            prog_ebind = prog_ebind.to(u.erg).value

            # Gather the results to the master
            ebind = comm.reduce(ebind, op=MPI.SUM, root=0)
            prog_ebind = comm.reduce(prog_ebind, op=MPI.SUM, root=0)

            if rank == 0:

                # Calculate stellar radii
                star_rs = np.sqrt((this_s_pos[:, 0] - cop[0]) ** 2
                                  + (this_s_pos[:, 1] - cop[1]) ** 2
                                  + (this_s_pos[:, 2] - cop[2]) ** 2)
                prog_star_rs = np.sqrt((prog_this_s_pos[:, 0] - prog_cop[0]) ** 2
                                       + (prog_this_s_pos[:,
                                                          1] - prog_cop[1]) ** 2
                                       + (prog_this_s_pos[:, 2] - prog_cop[2]) ** 2)
                okinds = star_rs < 0.03
                prog_okinds = prog_star_rs < 0.03

                # Include these results for plotting
                tot_hmrs.append(hmr)
                tot_prog_hmrs.extend(prog_hmr)
                feedback_energy.append(
                    np.sum(1.74 * 10 ** 49 * this_ini_ms[okinds]))
                prog_feedback_energy.append(np.sum(1.74 * 10 ** 49 *
                                                   prog_this_ini_ms[prog_okinds]))
                binding_energy.append(ebind)
                prog_binding_energy.append(prog_ebind)
                w.append(ws[ind])

    hdf_master.close()

    if rank == 0:

        # Convert to arrays
        tot_hmrs = np.array(tot_hmrs)
        tot_prog_hmrs = np.array(tot_prog_hmrs)
        feedback_energy = np.array(feedback_energy)
        prog_feedback_energy = np.array(prog_feedback_energy)
        binding_energy = np.array(binding_energy)
        prog_binding_energy = np.array(prog_binding_energy)
        w = np.array(w)

        # Compute delta
        delta_hmr = tot_hmrs - tot_prog_hmrs
        delta_fb = feedback_energy / prog_feedback_energy
        delta_eb = binding_energy / prog_binding_energy
        delta_prog = prog_binding_energy / prog_feedback_energy
        delta_current = binding_energy / feedback_energy

        norm = cm.TwoSlopeNorm(vmin=delta_hmr.min(), vcenter=0,
                               vmax=delta_hmr.max())

        # Sort by decreasing size to overlay shrinking galaxies
        sinds = np.argsort(delta_hmr)[::-1]
        delta_hmr = delta_hmr[sinds]
        delta_fb = delta_fb[sinds]
        delta_eb = delta_eb[sinds]
        delta_prog = delta_prog[sinds]
        delta_current = delta_current[sinds]

        # Set up plot
        fig = plt.figure(figsize=(3.5, 3.5))
        ax = fig.add_subplot(111)
        ax.loglog()

        okinds = np.logical_and(delta_eb > 0, delta_fb > 0)

        # Plot the scatter
        # Plot the scatter
        im = ax.hexbin(delta_eb[okinds], delta_fb[okinds],  gridsize=50,
                       mincnt=np.min(w) - (0.1 * np.min(w)),
                       C=delta_hmr[okinds], xscale="log", yscale="log",
                       reduce_C_function=np.mean, norm=norm,
                       linewidths=0.2, cmap="coolwarm")

        # Axes labels
        ax.set_xlabel("$E_\mathrm{fb}^\mathrm{B} / E_\mathrm{fb}^\mathrm{A}$")
        ax.set_ylabel(
            "$E_\mathrm{bind}^\mathrm{B} / E_\mathrm{bind}^\mathrm{A}$")

        cbar = fig.colorbar(im)
        cbar.set_label("$\Delta R_{1/2} / [\mathrm{pkpc}]$")

        # Save figure
        mkdir("plots/graph/")
        fig.savefig("plots/graph/delta_hmr_bind.png",
                    bbox_inches="tight")
        plt.close(fig)

        # Set up plot
        fig = plt.figure(figsize=(3.5, 3.5))
        ax = fig.add_subplot(111)
        ax.loglog()

        okinds = np.logical_and(delta_prog > 0, delta_current > 0)

        # Plot the scatter
        im = ax.hexbin(delta_prog[okinds], delta_current[okinds],  gridsize=50,
                       mincnt=np.min(w) - (0.1 * np.min(w)),
                       C=delta_hmr[okinds], xscale="log", yscale="log",
                       reduce_C_function=np.mean, norm=norm,
                       linewidths=0.2, cmap="coolwarm")

        # Axes labels
        ax.set_xlabel(
            "$E_\mathrm{bind}^\mathrm{A} / E_\mathrm{fb}^\mathrm{A}$")
        ax.set_ylabel(
            "$E_\mathrm{bind}^\mathrm{B} / E_\mathrm{fb}^\mathrm{B}$")

        cbar = fig.colorbar(im)
        cbar.set_label("$\Delta R_{1/2} / [\mathrm{pkpc}]$")

        # Save figure
        mkdir("plots/graph/")
        fig.savefig("plots/graph/delta_hmr_bind_fbratio.png",
                    bbox_inches="tight")
        plt.close(fig)


def plot_size_mass_evo_grid(stellar_data, snaps):

    # Define paths
    path = "/cosma/home/dp004/dc-rope1/cosma7/FLARES/flares-mergergraph/"
    halo_base = path + "data/halos/MEGAFLARES_halos_<reg>_<snap>.hdf5"
    graph_base = path + "data/dgraph/MEGAFLARES_graph_<reg>_<snap>.hdf5"

    # Set up the dictionary to store the graph information
    graph = {}

    # Define root snapshot
    root_snap = snaps[-1]

    # Extract galaxy data from the sizes dict for the root snap
    root_hmrs = stellar_data[root_snap]["HMRs"][...]
    root_mass = stellar_data[root_snap]["mass"][...]
    root_grps = stellar_data[root_snap]["Galaxy,GroupNumber"][...]
    root_subgrps = stellar_data[root_snap]["Galaxy,SubGroupNumber"][...]
    root_regions = stellar_data[root_snap]["regions"][...]

    # Define redshift norm
    norm = cm.Normalize(vmin=5, vmax=12)

    # Create data dictionary to speed up walking
    mega_data = {}
    for reg_int in np.unique(root_regions):
        if reg_int == 18:
            continue
        reg = str(reg_int).zfill(2)
        mega_data[reg] = {}
        for snap in snaps:
            mega_data[reg][snap] = {}

            # Open this new region
            this_halo_base = halo_base.replace("<reg>", reg)
            this_halo_base = this_halo_base.replace("<snap>", snap)
            this_graph_base = graph_base.replace("<reg>", reg)
            this_graph_base = this_graph_base.replace("<snap>", snap)
            hdf_halo = h5py.File(this_halo_base, "r")
            hdf_graph = h5py.File(this_graph_base, "r")

            # Get the MEGA ID arrays for both snapshots
            mega_data[reg][snap]["group_number"] = hdf_halo["group_number"][...]
            mega_data[reg][snap]["subgroup_number"] = hdf_halo["subgroup_number"][...]

            # Get the progenitor information
            mega_data[reg][snap]["ProgHaloIDs"] = hdf_graph["ProgHaloIDs"][...]
            mega_data[reg][snap]["prog_start_index"] = hdf_graph["prog_start_index"][...]

            hdf_halo.close()
            hdf_graph.close()

    # Loop over galaxies and populate the root level of the graph with
    # only the compact galaxies
    for ind in range(len(root_hmrs)):

        # Skip if the galaxy isn't compact
        if root_hmrs[ind] > 1:
            continue

        # Get ID
        g, sg = root_grps[ind], root_subgrps[ind]

        # Make an entry in the dict for it
        graph[(g, sg, ind)] = {"HMRs": [], "Masses": [], "z": []}

    # Loop over these root galaxies and populate the rest of the graph
    i = 0
    for key in graph:

        print("Walking %d (%d, %d, %d) of %d" % (i, key[0], key[1],
                                                 key[2], len(graph)), end="\r")

        # Extract IDs
        g, sg, ind = key

        # Get the region for this galaxy
        reg_int = root_regions[ind]
        if reg_int == 18:
            continue
        reg = str(reg_int).zfill(2)

        # Set up the snapshot
        snap_ind = len(snaps) - 1
        snap = root_snap
        prog_snap = snaps[snap_ind - 1]

        # Set up looping variables
        this_g, this_sg, this_ind = g, sg, ind

        i += 1

        # Loop until we don't have a progenitor to step to
        while snap_ind >= 0:

            # Get redshift
            z = float(snap.split("z")[-1].replace("p", "."))

            # Extract galaxy data from the sizes dict
            hmrs = stellar_data[snap]["HMRs"][...]
            mass = stellar_data[snap]["mass"][...]
            prog_grps = stellar_data[prog_snap]["Galaxy,GroupNumber"][...]
            prog_subgrps = stellar_data[prog_snap]["Galaxy,SubGroupNumber"][...]
            prog_regions = stellar_data[prog_snap]["regions"][...]

            # Put this galaxy in the graph
            if snap == root_snap:
                graph[(g, sg, ind)]["HMRs"].append(
                    hmrs[this_ind]  # / stellar_data[root_snap]["HMRs"][ind]
                )
                graph[(g, sg, ind)]["Masses"].append(
                    mass[this_ind]
                )
            else:
                graph[(g, sg, ind)]["HMRs"].extend(
                    hmrs[this_ind]  # / stellar_data[root_snap]["HMRs"][ind]
                )
                graph[(g, sg, ind)]["Masses"].extend(
                    mass[this_ind]
                )
            graph[(g, sg, ind)]["z"].append(z)

            # Get the MEGA ID arrays for both snapshots
            mega_grps = mega_data[reg][snap]["group_number"]
            mega_subgrps = mega_data[reg][snap]["subgroup_number"]
            mega_prog_grps = mega_data[reg][prog_snap]["group_number"]
            mega_prog_subgrps = mega_data[reg][prog_snap]["subgroup_number"]

            # Get the progenitor information
            prog_ids = mega_data[reg][snap]["ProgHaloIDs"]
            start_index = mega_data[reg][snap]["prog_start_index"]

            # Whats the MEGA ID of this galaxy?
            mega_ind = np.where(np.logical_and(mega_grps == this_g,
                                               mega_subgrps == this_sg))[0]

            # Get the progenitor
            start = start_index[mega_ind][0]
            main_prog = prog_ids[start]

            if start != 2**30:

                # Get this progenitors group and subgroup ID
                this_g = mega_prog_grps[main_prog]
                this_sg = mega_prog_subgrps[main_prog]

                # Get this progenitors index
                this_ind = np.where(
                    np.logical_and(prog_regions == reg_int,
                                   np.logical_and(prog_grps == this_g,
                                                  prog_subgrps == this_sg)))[0]

                if this_ind.size == 0:
                    break

                # Set snapshots
                snap_ind -= 1
                snap = snaps[snap_ind]
                prog_snap = snaps[snap_ind - 1]

            else:
                break

    # Set up plot parameters
    ylims = (10**-1.3, 10**1.3)
    xlims = (10**8, 10**11.5)

    # Define mins and maxs for binning
    min_hmr = np.inf
    max_hmr = 0

    # Get the max size reached in each main branch
    max_size = {}
    for key in list(graph.keys()):
        if len(graph[key]["HMRs"]) > 1:
            max_size[key] = graph[key]["HMRs"][-1]
            if max_size[key] > max_hmr:
                max_hmr = max_size[key]
            if max_size[key] < min_hmr:
                min_hmr = max_size[key]
        else:
            del graph[key]

    # Create array of max sizes to bin
    keys = list(graph.keys())
    max_size_arr = np.zeros(len(keys))
    max_mass_arr = np.zeros(len(keys))
    for ind, key in enumerate(keys):
        max_size_arr[ind] = max_size[key]
        max_mass_arr[ind] = graph[key]["Masses"][0]

    # Define size bins
    size_bin_edges = np.logspace(
        np.log10(np.min(max_size_arr) - 0.01 * np.min(max_size_arr)),
        np.log10(np.max(max_size_arr) + 0.01 * np.max(max_size_arr)),
        5)
    mass_bin_edges = [10**9, 10**9.5, 10**10, np.max(max_mass_arr) + 10**9]
    size_bins = np.digitize(max_size_arr, size_bin_edges) - 1
    mass_bins = np.digitize(max_mass_arr, mass_bin_edges) - 1

    print(np.unique(size_bins, return_counts=True))
    print(np.unique(mass_bins, return_counts=True))

    # Define plot grid shape
    nrows = len(size_bin_edges) - 1
    ncols = len(mass_bin_edges) - 1

    # Set up plot
    fig = plt.figure(figsize=(3.5 * ncols + 0.1 * 3.5, 3.5 * nrows))
    gs = gridspec.GridSpec(nrows=nrows, ncols=ncols + 1,
                           width_ratios=[20, ] * ncols + [1])
    gs.update(wspace=0.0, hspace=0.0)
    axes = np.empty((nrows, ncols), dtype=object)
    cax = fig.add_subplot(gs[:, -1])

    cax.tick_params(axis="both", top=False, bottom=False,
                    left=False, right=False,
                    labeltop=False, labelbottom=False,
                    labelleft=False, labelright=False)

    for i in range(nrows):
        for j in range(ncols):
            axes[i, j] = fig.add_subplot(gs[i, j])
            axes[i, j].loglog()
            axes[i, j].set_xlim(xlims)
            axes[i, j].set_ylim(ylims)
            if i < nrows - 1:
                axes[i, j].tick_params(axis='x', top=False, bottom=False,
                                       labeltop=False, labelbottom=False)
            if j > 0:
                axes[i, j].tick_params(axis='y', left=False, right=False,
                                       labelleft=False, labelright=False)

            if i == 0:
                axes[i, j].set_title("$%.1f \leq \log_{10}(M_\star^{z=5} / M_\odot) < %.1f$"
                                     % (np.log10(mass_bin_edges[j]),
                                        np.log10(mass_bin_edges[j + 1])))

            if j == 0:
                axes[i, j].annotate(
                    "$%.1f \leq R_{1/2}^{\mathrm{form}} / [\mathrm{pkpc}] < %.1f$"
                    % (size_bin_edges[i], size_bin_edges[i + 1]),
                    xy=(0, 0.5), xytext=(-axes[i, j].yaxis.labelpad - 5, 0),
                    xycoords=axes[i,
                                  j].yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center', rotation=90)

    # Loop over graphs
    ii = 0
    for ind, key in enumerate(keys):

        print("Plotting %d (%d, %d, %d) of %d" % (ii, key[0], key[1],
                                                  key[2], len(graph)), end="\r")

        i, j = size_bins[ind], mass_bins[ind]

        # Skip galaxies that fell outside the range
        if i == -1 or j == -1:
            continue

        # Plot the scatter
        im = axes[i, j].plot(graph[key]["Masses"], graph[key]["HMRs"],
                             color="grey", alpha=0.2, zorder=0)
        ii += 1

    # Loop over graphs
    ii = 0
    for ind, key in enumerate(keys):

        print("Plotting %d (%d, %d, %d) of %d" % (ii, key[0], key[1],
                                                  key[2], len(graph)), end="\r")

        i, j = size_bins[ind], mass_bins[ind]

        # Skip galaxies that fell outside the range
        if i == -1 or j == -1:
            continue

        # Plot the scatter
        im = axes[i, j].scatter(graph[key]["Masses"], graph[key]["HMRs"],
                                marker=".", edgecolors="none", s=20,
                                c=graph[key]["z"], cmap="cmr.chroma",
                                alpha=0.8, zorder=1, norm=norm)
        ii += 1

    # Axes labels
    for ax in axes[:, 0]:
        ax.set_ylabel("$R_{1/2\star} / [\mathrm{pkpc}]$")
    for ax in axes[-1, :]:
        ax.set_xlabel("$M_{\star} / M_{\odot}$")

    cbar = fig.colorbar(im, cax)
    cbar.set_label("$z$")

    # Save figure
    mkdir("plots/graph_plot/")
    fig.savefig("plots/graph_plot/size_mass_evo_all.png",
                bbox_inches="tight")
    plt.close(fig)


def plot_gas_size_mass_evo_grid(stellar_data, gas_data, snaps):

    # Define paths
    path = "/cosma/home/dp004/dc-rope1/cosma7/FLARES/flares-mergergraph/"
    halo_base = path + "data/halos/MEGAFLARES_halos_<reg>_<snap>.hdf5"
    graph_base = path + "data/dgraph/MEGAFLARES_graph_<reg>_<snap>.hdf5"

    # Set up the dictionary to store the graph information
    graph = {}

    # Define root snapshot
    root_snap = snaps[-1]

    # Extract galaxy data from the sizes dict for the root snap
    root_hmrs = stellar_data[root_snap]["HMRs"][...]
    root_mass = stellar_data[root_snap]["mass"][...]
    root_grps = stellar_data[root_snap]["Galaxy,GroupNumber"][...]
    root_subgrps = stellar_data[root_snap]["Galaxy,SubGroupNumber"][...]
    root_regions = stellar_data[root_snap]["regions"][...]

    # Define redshift norm
    norm = cm.Normalize(vmin=5, vmax=12)

    # Create data dictionary to speed up walking
    mega_data = {}
    for reg_int in np.unique(root_regions):
        if reg_int == 18:
            continue
        reg = str(reg_int).zfill(2)
        mega_data[reg] = {}
        for snap in snaps:
            mega_data[reg][snap] = {}

            # Open this new region
            this_halo_base = halo_base.replace("<reg>", reg)
            this_halo_base = this_halo_base.replace("<snap>", snap)
            this_graph_base = graph_base.replace("<reg>", reg)
            this_graph_base = this_graph_base.replace("<snap>", snap)
            hdf_halo = h5py.File(this_halo_base, "r")
            hdf_graph = h5py.File(this_graph_base, "r")

            # Get the MEGA ID arrays for both snapshots
            mega_data[reg][snap]["group_number"] = hdf_halo["group_number"][...]
            mega_data[reg][snap]["subgroup_number"] = hdf_halo["subgroup_number"][...]

            # Get the progenitor information
            mega_data[reg][snap]["ProgHaloIDs"] = hdf_graph["ProgHaloIDs"][...]
            mega_data[reg][snap]["prog_start_index"] = hdf_graph["prog_start_index"][...]

            hdf_halo.close()
            hdf_graph.close()

    # Loop over galaxies and populate the root level of the graph with
    # only the compact galaxies
    for ind in range(len(root_hmrs)):

        # Skip if the galaxy isn't compact
        if root_hmrs[ind] > 1:
            continue

        # Get ID
        g, sg = root_grps[ind], root_subgrps[ind]

        # Make an entry in the dict for it
        graph[(g, sg, ind)] = {"HMRs": [], "gHMRs": [], "Masses": [], "z": []}

    # Loop over these root galaxies and populate the rest of the graph
    i = 0
    for key in graph:

        print("Walking %d (%d, %d, %d) of %d" % (i, key[0], key[1],
                                                 key[2], len(graph)), end="\r")

        # Extract IDs
        g, sg, ind = key

        # Get the region for this galaxy
        reg_int = root_regions[ind]
        if reg_int == 18:
            continue
        reg = str(reg_int).zfill(2)

        # Set up the snapshot
        snap_ind = len(snaps) - 1
        snap = root_snap
        prog_snap = snaps[snap_ind - 1]

        # Set up looping variables
        this_g, this_sg, this_ind = g, sg, ind

        i += 1

        # Loop until we don't have a progenitor to step to
        while snap_ind >= 0:

            # Get redshift
            z = float(snap.split("z")[-1].replace("p", "."))

            # Extract galaxy data from the sizes dict
            ghmrs = gas_data[snap]["HMRs"][...]
            hmrs = stellar_data[snap]["HMRs"][...]
            mass = stellar_data[snap]["mass"][...]
            prog_grps = stellar_data[prog_snap]["Galaxy,GroupNumber"][...]
            prog_subgrps = stellar_data[prog_snap]["Galaxy,SubGroupNumber"][...]
            prog_regions = stellar_data[prog_snap]["regions"][...]

            # Put this galaxy in the graph
            if snap == root_snap:
                graph[(g, sg, ind)]["HMRs"].append(
                    hmrs[this_ind]  # / stellar_data[root_snap]["HMRs"][ind]
                )
                graph[(g, sg, ind)]["gHMRs"].append(
                    ghmrs[this_ind]  # / stellar_data[root_snap]["HMRs"][ind]
                )
                graph[(g, sg, ind)]["Masses"].append(
                    mass[this_ind]
                )
            else:
                graph[(g, sg, ind)]["HMRs"].extend(
                    hmrs[this_ind]  # / stellar_data[root_snap]["HMRs"][ind]
                )
                graph[(g, sg, ind)]["gHMRs"].extend(
                    ghmrs[this_ind]  # / stellar_data[root_snap]["HMRs"][ind]
                )
                graph[(g, sg, ind)]["Masses"].extend(
                    mass[this_ind]
                )
            graph[(g, sg, ind)]["z"].append(z)

            # Get the MEGA ID arrays for both snapshots
            mega_grps = mega_data[reg][snap]["group_number"]
            mega_subgrps = mega_data[reg][snap]["subgroup_number"]
            mega_prog_grps = mega_data[reg][prog_snap]["group_number"]
            mega_prog_subgrps = mega_data[reg][prog_snap]["subgroup_number"]

            # Get the progenitor information
            prog_ids = mega_data[reg][snap]["ProgHaloIDs"]
            start_index = mega_data[reg][snap]["prog_start_index"]

            # Whats the MEGA ID of this galaxy?
            mega_ind = np.where(np.logical_and(mega_grps == this_g,
                                               mega_subgrps == this_sg))[0]

            # Get the progenitor
            start = start_index[mega_ind][0]
            main_prog = prog_ids[start]

            if start != 2**30:

                # Get this progenitors group and subgroup ID
                this_g = mega_prog_grps[main_prog]
                this_sg = mega_prog_subgrps[main_prog]

                # Get this progenitors index
                this_ind = np.where(
                    np.logical_and(prog_regions == reg_int,
                                   np.logical_and(prog_grps == this_g,
                                                  prog_subgrps == this_sg)))[0]

                if this_ind.size == 0:
                    break

                # Set snapshots
                snap_ind -= 1
                snap = snaps[snap_ind]
                prog_snap = snaps[snap_ind - 1]

            else:
                break

    # Set up plot parameters
    ylims = (10**-1.5, 10**1.5)
    xlims = (10**8, 10**11.5)

    # Define mins and maxs for binning
    min_hmr = np.inf
    max_hmr = 0

    # Get the max size reached in each main branch
    max_size = {}
    for key in list(graph.keys()):
        if len(graph[key]["HMRs"]) > 1:
            max_size[key] = graph[key]["HMRs"][-1]
            if max_size[key] > max_hmr:
                max_hmr = max_size[key]
            if max_size[key] < min_hmr:
                min_hmr = max_size[key]
        else:
            del graph[key]

    # Create array of max sizes to bin
    keys = list(graph.keys())
    max_size_arr = np.zeros(len(keys))
    max_mass_arr = np.zeros(len(keys))
    for ind, key in enumerate(keys):
        max_size_arr[ind] = max_size[key]
        max_mass_arr[ind] = graph[key]["Masses"][0]

    # Define size bins
    size_bin_edges = np.logspace(
        np.log10(np.min(max_size_arr) - 0.01 * np.min(max_size_arr)),
        np.log10(np.max(max_size_arr) + 0.01 * np.max(max_size_arr)),
        5)
    mass_bin_edges = [10**9, 10**9.5, 10**10, np.max(max_mass_arr) + 10**9]
    size_bins = np.digitize(max_size_arr, size_bin_edges) - 1
    mass_bins = np.digitize(max_mass_arr, mass_bin_edges) - 1

    print(np.unique(size_bins, return_counts=True))
    print(np.unique(mass_bins, return_counts=True))

    # Define plot grid shape
    nrows = len(size_bin_edges) - 1
    ncols = len(mass_bin_edges) - 1

    # Set up plot
    fig = plt.figure(figsize=(3.5 * ncols + 0.1 * 3.5, 3.5 * nrows))
    gs = gridspec.GridSpec(nrows=nrows, ncols=ncols + 1,
                           width_ratios=[20, ] * ncols + [1])
    gs.update(wspace=0.0, hspace=0.0)
    axes = np.empty((nrows, ncols), dtype=object)
    cax = fig.add_subplot(gs[:, -1])

    cax.tick_params(axis="both", top=False, bottom=False,
                    left=False, right=False,
                    labeltop=False, labelbottom=False,
                    labelleft=False, labelright=False)

    for i in range(nrows):
        for j in range(ncols):
            axes[i, j] = fig.add_subplot(gs[i, j])
            axes[i, j].loglog()
            axes[i, j].set_xlim(xlims)
            axes[i, j].set_ylim(ylims)
            if i < nrows - 1:
                axes[i, j].tick_params(axis='x', top=False, bottom=False,
                                       labeltop=False, labelbottom=False)
            if j > 0:
                axes[i, j].tick_params(axis='y', left=False, right=False,
                                       labelleft=False, labelright=False)

            if i == 0:
                axes[i, j].set_title("$%.1f \leq \log_{10}(M_\star^{z=5} / M_\odot) < %.1f$"
                                     % (np.log10(mass_bin_edges[j]),
                                        np.log10(mass_bin_edges[j + 1])))

            if j == 0:
                axes[i, j].annotate(
                    "$%.1f \leq R_{1/2}^{\mathrm{form}} / [\mathrm{pkpc}] < %.1f$"
                    % (size_bin_edges[i], size_bin_edges[i + 1]),
                    xy=(0, 0.5), xytext=(-axes[i, j].yaxis.labelpad - 5, 0),
                    xycoords=axes[i,
                                  j].yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center', rotation=90)

    # Loop over graphs
    ii = 0
    for ind, key in enumerate(keys):

        print("Plotting %d (%d, %d, %d) of %d" % (ii, key[0], key[1],
                                                  key[2], len(graph)), end="\r")

        i, j = size_bins[ind], mass_bins[ind]

        # Skip galaxies that fell outside the range
        if i == -1 or j == -1:
            continue

        # Plot the scatter
        im = axes[i, j].plot(graph[key]["Masses"], graph[key]["gHMRs"],
                             color="grey", alpha=0.2, zorder=0)
        ii += 1

    # Loop over graphs
    ii = 0
    for ind, key in enumerate(keys):

        print("Plotting %d (%d, %d, %d) of %d" % (ii, key[0], key[1],
                                                  key[2], len(graph)), end="\r")

        i, j = size_bins[ind], mass_bins[ind]

        # Skip galaxies that fell outside the range
        if i == -1 or j == -1:
            continue

        # Plot the scatter
        im = axes[i, j].scatter(graph[key]["Masses"], graph[key]["gHMRs"],
                                marker=".", edgecolors="none", s=20,
                                c=graph[key]["z"], cmap="cmr.chroma",
                                alpha=0.8, zorder=1, norm=norm)
        ii += 1

    # Axes labels
    for ax in axes[:, 0]:
        ax.set_ylabel("$R_{1/2}^\mathrm{gas} / [\mathrm{pkpc}]$")
    for ax in axes[-1, :]:
        ax.set_xlabel("$M_{\star} / M_{\odot}$")

    cbar = fig.colorbar(im, cax)
    cbar.set_label("$z$")

    # Save figure
    mkdir("plots/graph_plot/")
    fig.savefig("plots/graph_plot/gas_size_mass_evo_all.png",
                bbox_inches="tight")
    plt.close(fig)


def plot_gas_size_gasmass_evo_grid(stellar_data, gas_data, snaps):

    # Define paths
    path = "/cosma/home/dp004/dc-rope1/cosma7/FLARES/flares-mergergraph/"
    halo_base = path + "data/halos/MEGAFLARES_halos_<reg>_<snap>.hdf5"
    graph_base = path + "data/dgraph/MEGAFLARES_graph_<reg>_<snap>.hdf5"

    # Set up the dictionary to store the graph information
    graph = {}

    # Define root snapshot
    root_snap = snaps[-1]

    # Extract galaxy data from the sizes dict for the root snap
    root_hmrs = stellar_data[root_snap]["HMRs"][...]
    root_mass = stellar_data[root_snap]["mass"][...]
    root_grps = stellar_data[root_snap]["Galaxy,GroupNumber"][...]
    root_subgrps = stellar_data[root_snap]["Galaxy,SubGroupNumber"][...]
    root_regions = stellar_data[root_snap]["regions"][...]

    # Define redshift norm
    norm = cm.Normalize(vmin=5, vmax=12)

    # Create data dictionary to speed up walking
    mega_data = {}
    for reg_int in np.unique(root_regions):
        if reg_int == 18:
            continue
        reg = str(reg_int).zfill(2)
        mega_data[reg] = {}
        for snap in snaps:
            mega_data[reg][snap] = {}

            # Open this new region
            this_halo_base = halo_base.replace("<reg>", reg)
            this_halo_base = this_halo_base.replace("<snap>", snap)
            this_graph_base = graph_base.replace("<reg>", reg)
            this_graph_base = this_graph_base.replace("<snap>", snap)
            hdf_halo = h5py.File(this_halo_base, "r")
            hdf_graph = h5py.File(this_graph_base, "r")

            # Get the MEGA ID arrays for both snapshots
            mega_data[reg][snap]["group_number"] = hdf_halo["group_number"][...]
            mega_data[reg][snap]["subgroup_number"] = hdf_halo["subgroup_number"][...]

            # Get the progenitor information
            mega_data[reg][snap]["ProgHaloIDs"] = hdf_graph["ProgHaloIDs"][...]
            mega_data[reg][snap]["prog_start_index"] = hdf_graph["prog_start_index"][...]

            hdf_halo.close()
            hdf_graph.close()

    # Loop over galaxies and populate the root level of the graph with
    # only the compact galaxies
    for ind in range(len(root_hmrs)):

        # Skip if the galaxy isn't compact
        if root_hmrs[ind] > 1:
            continue

        # Get ID
        g, sg = root_grps[ind], root_subgrps[ind]

        # Make an entry in the dict for it
        graph[(g, sg, ind)] = {"HMRs": [], "gHMRs": [], "Masses": [], "z": []}

    # Loop over these root galaxies and populate the rest of the graph
    i = 0
    for key in graph:

        print("Walking %d (%d, %d, %d) of %d" % (i, key[0], key[1],
                                                 key[2], len(graph)), end="\r")

        # Extract IDs
        g, sg, ind = key

        # Get the region for this galaxy
        reg_int = root_regions[ind]
        if reg_int == 18:
            continue
        reg = str(reg_int).zfill(2)

        # Set up the snapshot
        snap_ind = len(snaps) - 1
        snap = root_snap
        prog_snap = snaps[snap_ind - 1]

        # Set up looping variables
        this_g, this_sg, this_ind = g, sg, ind

        i += 1

        # Loop until we don't have a progenitor to step to
        while snap_ind >= 0:

            # Get redshift
            z = float(snap.split("z")[-1].replace("p", "."))

            # Extract galaxy data from the sizes dict
            ghmrs = gas_data[snap]["HMRs"][...]
            hmrs = stellar_data[snap]["HMRs"][...]
            mass = gas_data[snap]["mass"][...]
            prog_grps = stellar_data[prog_snap]["Galaxy,GroupNumber"][...]
            prog_subgrps = stellar_data[prog_snap]["Galaxy,SubGroupNumber"][...]
            prog_regions = stellar_data[prog_snap]["regions"][...]

            # Put this galaxy in the graph
            if snap == root_snap:
                graph[(g, sg, ind)]["HMRs"].append(
                    hmrs[this_ind]  # / stellar_data[root_snap]["HMRs"][ind]
                )
                graph[(g, sg, ind)]["gHMRs"].append(
                    ghmrs[this_ind]  # / stellar_data[root_snap]["HMRs"][ind]
                )
                graph[(g, sg, ind)]["Masses"].append(
                    mass[this_ind]
                )
            else:
                graph[(g, sg, ind)]["HMRs"].extend(
                    hmrs[this_ind]  # / stellar_data[root_snap]["HMRs"][ind]
                )
                graph[(g, sg, ind)]["gHMRs"].extend(
                    ghmrs[this_ind]  # / stellar_data[root_snap]["HMRs"][ind]
                )
                graph[(g, sg, ind)]["Masses"].extend(
                    mass[this_ind]
                )
            graph[(g, sg, ind)]["z"].append(z)

            # Get the MEGA ID arrays for both snapshots
            mega_grps = mega_data[reg][snap]["group_number"]
            mega_subgrps = mega_data[reg][snap]["subgroup_number"]
            mega_prog_grps = mega_data[reg][prog_snap]["group_number"]
            mega_prog_subgrps = mega_data[reg][prog_snap]["subgroup_number"]

            # Get the progenitor information
            prog_ids = mega_data[reg][snap]["ProgHaloIDs"]
            start_index = mega_data[reg][snap]["prog_start_index"]

            # Whats the MEGA ID of this galaxy?
            mega_ind = np.where(np.logical_and(mega_grps == this_g,
                                               mega_subgrps == this_sg))[0]

            # Get the progenitor
            start = start_index[mega_ind][0]
            main_prog = prog_ids[start]

            if start != 2**30:

                # Get this progenitors group and subgroup ID
                this_g = mega_prog_grps[main_prog]
                this_sg = mega_prog_subgrps[main_prog]

                # Get this progenitors index
                this_ind = np.where(
                    np.logical_and(prog_regions == reg_int,
                                   np.logical_and(prog_grps == this_g,
                                                  prog_subgrps == this_sg)))[0]

                if this_ind.size == 0:
                    break

                # Set snapshots
                snap_ind -= 1
                snap = snaps[snap_ind]
                prog_snap = snaps[snap_ind - 1]

            else:
                break

    # Set up plot parameters
    ylims = (10**-1.5, 10**1.5)
    xlims = (10**8, 10**11.5)

    # Define mins and maxs for binning
    min_hmr = np.inf
    max_hmr = 0

    # Get the max size reached in each main branch
    max_size = {}
    for key in list(graph.keys()):
        if len(graph[key]["HMRs"]) > 1:
            max_size[key] = graph[key]["HMRs"][-1]
            if max_size[key] > max_hmr:
                max_hmr = max_size[key]
            if max_size[key] < min_hmr:
                min_hmr = max_size[key]
        else:
            del graph[key]

    # Create array of max sizes to bin
    keys = list(graph.keys())
    max_size_arr = np.zeros(len(keys))
    max_mass_arr = np.zeros(len(keys))
    for ind, key in enumerate(keys):
        max_size_arr[ind] = max_size[key]
        max_mass_arr[ind] = graph[key]["Masses"][0]

    # Define size bins
    size_bin_edges = np.logspace(
        np.log10(np.min(max_size_arr) - 0.01 * np.min(max_size_arr)),
        np.log10(np.max(max_size_arr) + 0.01 * np.max(max_size_arr)),
        5)
    mass_bin_edges = [10**9, 10**9.5, 10**10, np.max(max_mass_arr) + 10**9]
    size_bins = np.digitize(max_size_arr, size_bin_edges) - 1
    mass_bins = np.digitize(max_mass_arr, mass_bin_edges) - 1

    print(np.unique(size_bins, return_counts=True))
    print(np.unique(mass_bins, return_counts=True))

    # Define plot grid shape
    nrows = len(size_bin_edges) - 1
    ncols = len(mass_bin_edges) - 1

    # Set up plot
    fig = plt.figure(figsize=(3.5 * ncols + 0.1 * 3.5, 3.5 * nrows))
    gs = gridspec.GridSpec(nrows=nrows, ncols=ncols + 1,
                           width_ratios=[20, ] * ncols + [1])
    gs.update(wspace=0.0, hspace=0.0)
    axes = np.empty((nrows, ncols), dtype=object)
    cax = fig.add_subplot(gs[:, -1])

    cax.tick_params(axis="both", top=False, bottom=False,
                    left=False, right=False,
                    labeltop=False, labelbottom=False,
                    labelleft=False, labelright=False)

    for i in range(nrows):
        for j in range(ncols):
            axes[i, j] = fig.add_subplot(gs[i, j])
            axes[i, j].loglog()
            axes[i, j].set_xlim(xlims)
            axes[i, j].set_ylim(ylims)
            if i < nrows - 1:
                axes[i, j].tick_params(axis='x', top=False, bottom=False,
                                       labeltop=False, labelbottom=False)
            if j > 0:
                axes[i, j].tick_params(axis='y', left=False, right=False,
                                       labelleft=False, labelright=False)

            if i == 0:
                axes[i, j].set_title("$%.1f \leq \log_{10}(M_\star^{z=5} / M_\odot) < %.1f$"
                                     % (np.log10(mass_bin_edges[j]),
                                        np.log10(mass_bin_edges[j + 1])))

            if j == 0:
                axes[i, j].annotate(
                    "$%.1f \leq R_{1/2}^{\mathrm{form}} / [\mathrm{pkpc}] < %.1f$"
                    % (size_bin_edges[i], size_bin_edges[i + 1]),
                    xy=(0, 0.5), xytext=(-axes[i, j].yaxis.labelpad - 5, 0),
                    xycoords=axes[i,
                                  j].yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center', rotation=90)

    # Loop over graphs
    ii = 0
    for ind, key in enumerate(keys):

        print("Plotting %d (%d, %d, %d) of %d" % (ii, key[0], key[1],
                                                  key[2], len(graph)), end="\r")

        i, j = size_bins[ind], mass_bins[ind]

        # Skip galaxies that fell outside the range
        if i == -1 or j == -1:
            continue

        # Plot the scatter
        im = axes[i, j].plot(graph[key]["Masses"], graph[key]["gHMRs"],
                             color="grey", alpha=0.2, zorder=0)
        ii += 1

    # Loop over graphs
    ii = 0
    for ind, key in enumerate(keys):

        print("Plotting %d (%d, %d, %d) of %d" % (ii, key[0], key[1],
                                                  key[2], len(graph)), end="\r")

        i, j = size_bins[ind], mass_bins[ind]

        # Skip galaxies that fell outside the range
        if i == -1 or j == -1:
            continue

        # Plot the scatter
        im = axes[i, j].scatter(graph[key]["Masses"], graph[key]["gHMRs"],
                                marker=".", edgecolors="none", s=20,
                                c=graph[key]["z"], cmap="cmr.chroma",
                                alpha=0.8, zorder=1, norm=norm)
        ii += 1

    # Axes labels
    for ax in axes[:, 0]:
        ax.set_ylabel("$R_{1/2}^\mathrm{gas} / [\mathrm{pkpc}]$")
    for ax in axes[-1, :]:
        ax.set_xlabel("$M_{\mathrm{gas}} / M_{\odot}$")

    cbar = fig.colorbar(im, cax)
    cbar.set_label("$z$")

    # Save figure
    mkdir("plots/graph_plot/")
    fig.savefig("plots/graph_plot/gas_size_gasmass_evo_all.png",
                bbox_inches="tight")
    plt.close(fig)


def plot_size_sfr_evo_grid(stellar_data, snaps):

    # Define paths
    path = "/cosma/home/dp004/dc-rope1/cosma7/FLARES/flares-mergergraph/"
    halo_base = path + "data/halos/MEGAFLARES_halos_<reg>_<snap>.hdf5"
    graph_base = path + "data/dgraph/MEGAFLARES_graph_<reg>_<snap>.hdf5"

    # Set up the dictionary to store the graph information
    graph = {}

    # Define root snapshot
    root_snap = snaps[-1]

    # Extract galaxy data from the sizes dict for the root snap
    root_hmrs = stellar_data[root_snap]["HMRs"][...]
    root_mass = stellar_data[root_snap]["mass"][...]
    root_grps = stellar_data[root_snap]["Galaxy,GroupNumber"][...]
    root_subgrps = stellar_data[root_snap]["Galaxy,SubGroupNumber"][...]
    root_regions = stellar_data[root_snap]["regions"][...]

    # Define redshift norm
    norm = cm.Normalize(vmin=5, vmax=12)

    # Create data dictionary to speed up walking
    mega_data = {}
    for reg_int in np.unique(root_regions):
        if reg_int == 18:
            continue
        reg = str(reg_int).zfill(2)
        mega_data[reg] = {}
        for snap in snaps:
            mega_data[reg][snap] = {}

            # Open this new region
            this_halo_base = halo_base.replace("<reg>", reg)
            this_halo_base = this_halo_base.replace("<snap>", snap)
            this_graph_base = graph_base.replace("<reg>", reg)
            this_graph_base = this_graph_base.replace("<snap>", snap)
            hdf_halo = h5py.File(this_halo_base, "r")
            hdf_graph = h5py.File(this_graph_base, "r")

            # Get the MEGA ID arrays for both snapshots
            mega_data[reg][snap]["group_number"] = hdf_halo["group_number"][...]
            mega_data[reg][snap]["subgroup_number"] = hdf_halo["subgroup_number"][...]

            # Get the progenitor information
            mega_data[reg][snap]["ProgHaloIDs"] = hdf_graph["ProgHaloIDs"][...]
            mega_data[reg][snap]["prog_start_index"] = hdf_graph["prog_start_index"][...]

            hdf_halo.close()
            hdf_graph.close()

    # Loop over galaxies and populate the root level of the graph with
    # only the compact galaxies
    for ind in range(len(root_hmrs)):

        # Skip if the galaxy isn't compact
        if root_hmrs[ind] > 1:
            continue

        # Get ID
        g, sg = root_grps[ind], root_subgrps[ind]

        # Make an entry in the dict for it
        graph[(g, sg, ind)] = {"HMRs": [], "Masses": [], "ssfr": []}

    # Loop over these root galaxies and populate the rest of the graph
    i = 0
    for key in graph:

        print("Walking %d (%d, %d, %d) of %d" % (i, key[0], key[1],
                                                 key[2], len(graph)), end="\r")

        # Extract IDs
        g, sg, ind = key

        # Get the region for this galaxy
        reg_int = root_regions[ind]
        if reg_int == 18:
            continue
        reg = str(reg_int).zfill(2)

        # Set up the snapshot
        snap_ind = len(snaps) - 1
        snap = root_snap
        prog_snap = snaps[snap_ind - 1]

        # Set up looping variables
        this_g, this_sg, this_ind = g, sg, ind

        i += 1

        # Loop until we don't have a progenitor to step to
        while snap_ind >= 0:

            # Get redshift
            z = float(snap.split("z")[-1].replace("p", "."))

            # Extract galaxy data from the sizes dict
            hmrs = stellar_data[snap]["HMRs"][...]
            mass = stellar_data[snap]["mass"][...]
            prog_grps = stellar_data[prog_snap]["Galaxy,GroupNumber"][...]
            prog_subgrps = stellar_data[prog_snap]["Galaxy,SubGroupNumber"][...]
            prog_regions = stellar_data[prog_snap]["regions"][...]
            ages = stellar_data[snap]["Particle,S_Age"][...]
            ini_ms = stellar_data[snap]["Particle,S_MassInitial"][...]
            begins = stellar_data[snap]["begin"][...]
            apps = stellar_data[snap]["Particle/Apertures/Star,30"][...]
            lengths = stellar_data[snap]["Galaxy,S_Length"][...]
            ms = stellar_data[snap]["Particle,S_Mass"][...]

            # Create boolean array identifying stars born in the last 100 Myrs
            # and are within the 30 pkpc aperture
            age_okinds = ages < 0.1
            okinds = np.logical_and(apps, age_okinds)

            # Put this galaxy in the graph
            if snap == root_snap:

                # Extract this galaxies data
                b = begins[this_ind]
                nstar = lengths[this_ind]
                app = apps[b: b + nstar]
                this_ini_ms = np.sum(
                    ini_ms[b: b + nstar][okinds[b: b + nstar]]) * 10 ** 10
                gal_m = np.sum(ms[b: b + nstar][app]) * 10 ** 10

                graph[(g, sg, ind)]["HMRs"].append(
                    hmrs[this_ind]  # / stellar_data[root_snap]["HMRs"][ind]
                )
                graph[(g, sg, ind)]["Masses"].append(
                    mass[this_ind]
                )
                graph[(g, sg, ind)]["ssfr"].append(this_ini_ms / 0.1
                                                   / gal_m)
            else:

                # Extract this galaxies data
                b = begins[this_ind][0]
                nstar = lengths[this_ind][0]
                app = apps[b: b + nstar]
                this_ini_ms = np.sum(
                    ini_ms[b: b + nstar][okinds[b: b + nstar]]) * 10 ** 10
                gal_m = np.sum(ms[b: b + nstar][app]) * 10 ** 10

                graph[(g, sg, ind)]["HMRs"].extend(
                    hmrs[this_ind]  # / stellar_data[root_snap]["HMRs"][ind]
                )
                graph[(g, sg, ind)]["Masses"].extend(
                    mass[this_ind]
                )
                graph[(g, sg, ind)]["ssfr"].append(this_ini_ms / 0.1
                                                   / gal_m)

            # Get the MEGA ID arrays for both snapshots
            mega_grps = mega_data[reg][snap]["group_number"]
            mega_subgrps = mega_data[reg][snap]["subgroup_number"]
            mega_prog_grps = mega_data[reg][prog_snap]["group_number"]
            mega_prog_subgrps = mega_data[reg][prog_snap]["subgroup_number"]

            # Get the progenitor information
            prog_ids = mega_data[reg][snap]["ProgHaloIDs"]
            start_index = mega_data[reg][snap]["prog_start_index"]

            # Whats the MEGA ID of this galaxy?
            mega_ind = np.where(np.logical_and(mega_grps == this_g,
                                               mega_subgrps == this_sg))[0]

            # Get the progenitor
            start = start_index[mega_ind][0]
            main_prog = prog_ids[start]

            if start != 2**30:

                # Get this progenitors group and subgroup ID
                this_g = mega_prog_grps[main_prog]
                this_sg = mega_prog_subgrps[main_prog]

                # Get this progenitors index
                this_ind = np.where(
                    np.logical_and(prog_regions == reg_int,
                                   np.logical_and(prog_grps == this_g,
                                                  prog_subgrps == this_sg)))[0]

                if this_ind.size == 0:
                    break

                # Set snapshots
                snap_ind -= 1
                snap = snaps[snap_ind]
                prog_snap = snaps[snap_ind - 1]

            else:
                break

    # Set up plot parameters
    ylims = (10**-1.3, 10**1.3)
    xlims = (10**8, 10**11.5)

    # Define mins and maxs for binning
    min_hmr = np.inf
    max_hmr = 0

    # Get the max size reached in each main branch
    max_size = {}
    for key in list(graph.keys()):
        if len(graph[key]["HMRs"]) > 1:
            max_size[key] = graph[key]["HMRs"][0] - np.max(graph[key]["HMRs"])
            if max_size[key] > max_hmr:
                max_hmr = max_size[key]
            if max_size[key] < min_hmr:
                min_hmr = max_size[key]
        else:
            del graph[key]

    # Define size bins
    size_bins = np.linspace(-1.5, 0.2, 9)

    # Define plot grid shape
    nrows = int((len(size_bins) - 1) / 2)
    ncols = 2

    # Set up plot
    fig = plt.figure(figsize=(3.5 * ncols + 0.1 * 3.5, 3.5 * nrows))
    gs = gridspec.GridSpec(nrows=nrows, ncols=ncols + 1,
                           width_ratios=[20, ] * ncols + [1])
    gs.update(wspace=0.0, hspace=0.0)
    axes = np.empty((nrows, ncols), dtype=object)
    cax = fig.add_subplot(gs[:, -1])

    cax.tick_params(axis="both", top=False, bottom=False,
                    left=False, right=False,
                    labeltop=False, labelbottom=False,
                    labelleft=False, labelright=False)

    # Create grid bin reference
    grid = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1)]

    for k in range(len(size_bins) - 1):
        i, j = grid[k]
        axes[i, j] = fig.add_subplot(gs[i, j])
        axes[i, j].loglog()
        axes[i, j].set_xlim(xlims)
        axes[i, j].set_ylim(ylims)
        if i < nrows - 1:
            axes[i, j].tick_params(axis='x', top=False, bottom=False,
                                   labeltop=False, labelbottom=False)
        if j > 0:
            axes[i, j].tick_params(axis='y', left=False, right=False,
                                   labelleft=False, labelright=False)

    for k in range(len(size_bins) - 1):
        i, j = grid[k]

        # Loop over graphs
        ii = 0
        for key in graph:

            print("Plotting %d (%d, %d, %d) of %d" % (ii, key[0], key[1],
                                                      key[2], len(graph)), end="\r")

            if size_bins[k] <= max_size[key] and size_bins[k + 1] > max_size[key]:

                # Plot the scatter
                im = axes[i, j].plot(graph[key]["Masses"], graph[key]["HMRs"],
                                     color="grey", alpha=0.2, zorder=0)
            ii += 1

        # Loop over graphs
        ii = 0
        for key in graph:

            print("Plotting %d (%d, %d, %d) of %d" % (ii, key[0], key[1],
                                                      key[2], len(graph)), end="\r")

            if size_bins[k] <= max_size[key] and size_bins[k + 1] > max_size[key]:

                # Plot the scatter
                im = axes[i, j].scatter(graph[key]["Masses"], graph[key]["HMRs"],
                                        marker=".", edgecolors="none", s=20,
                                        c=graph[key]["ssfr"], cmap="cmr.chroma",
                                        alpha=0.8, zorder=1, norm=cm.LogNorm(vmin=10**-2.1, vmax=10**1.1))
            ii += 1

    # Axes labels
    for ax in axes[:, 0]:
        ax.set_ylabel("$R_{1/2\star} / [\mathrm{pkpc}]$")
    for ax in axes[-1, :]:
        ax.set_xlabel("$M_{\star} / M_{\odot}$")

    cbar = fig.colorbar(im, cax)
    cbar.set_label("$\mathrm{sSFR} / [\mathrm{Gyr}^{-1}]$")

    # Save figure
    mkdir("plots/graph_plot/")
    fig.savefig("plots/graph_plot/size_ssfr_evo_all.png",
                bbox_inches="tight")
    plt.close(fig)


def plot_ssfr_mass_size_change(stellar_data, gas_data, snaps, weight_norm):

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
    tot_hmrs_gas = []
    tot_prog_hmrs_gas = []
    tot_mass = []
    tot_ssfr = []
    no_prog_mass = []
    no_prog_ssfr = []
    w = []

    # Physical constants
    G = (const.G.to(u.Mpc ** 3 * u.M_sun ** -1 * u.yr ** -2))

    # Loop over snapshots
    for snap, prog_snap in zip(current_snaps, prog_snaps):

        if snap != current_snaps[-1]:
            continue

        print(snap, prog_snap)

        # Get redshift
        z = float(snap.split("z")[-1].replace("p", "."))

        # Define softening length
        if z <= 2.8:
            soft = 0.000474390 / 0.6777
        else:
            soft = 0.001802390 / (0.6777 * (1 + z))

        # Open fake region
        reg = "100"
        reg_int = -1

        # Extract galaxy data from the sizes dict
        hmrs = stellar_data[snap]["HMRs"][...]
        ghmrs = gas_data[snap]["HMRs"][...]
        print("There are %d galaxies" % len(hmrs))
        print("There are %d compact galaxies" % len(hmrs[hmrs < 1]))
        prog_hmrs = stellar_data[prog_snap]["HMRs"][...]
        prog_ghmrs = gas_data[prog_snap]["HMRs"][...]
        grps = stellar_data[snap]["Galaxy,GroupNumber"][...]
        subgrps = stellar_data[snap]["Galaxy,SubGroupNumber"][...]
        prog_grps = stellar_data[prog_snap]["Galaxy,GroupNumber"][...]
        prog_subgrps = stellar_data[prog_snap]["Galaxy,SubGroupNumber"][...]
        regions = stellar_data[snap]["regions"][...]
        ws = stellar_data[snap]["weights"][...]
        prog_regions = stellar_data[prog_snap]["regions"][...]
        ages = stellar_data[snap]["Particle,S_Age"][...] * 1000
        ini_ms = stellar_data[snap]["Particle,S_MassInitial"][...] * 10 ** 10
        begins = stellar_data[snap]["begin"][...]
        apps = stellar_data[snap]["Particle/Apertures/Star,30"][...]
        lengths = stellar_data[snap]["Galaxy,S_Length"][...]
        ms = stellar_data[snap]["Particle,S_Mass"][...]

        # Create boolean array identifying stars born in the last 100 Myrs
        # and are within the 30 pkpc aperture
        age_okinds = ages < 100
        okinds = np.logical_and(apps, age_okinds)

        # Loop over galaxies
        for ind in range(len(hmrs)):

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
                mega_prog_grps = hdf_prog["group_number"][...]
                mega_prog_subgrps = hdf_prog["subgroup_number"][...]

                # Get the progenitor information
                prog_ids = hdf_graph["ProgHaloIDs"][...]
                start_index = hdf_graph["prog_start_index"][...]
                nprogs = hdf_graph["n_progs"][...]

                hdf_halo.close()
                hdf_prog.close()
                hdf_graph.close()

            # Extract this galaxies data
            b = begins[ind]
            nstar = lengths[ind]
            app = apps[b: b + nstar]
            this_ini_ms = np.sum(ini_ms[b: b + nstar][okinds[b: b + nstar]])
            gal_m = np.sum(ms[b: b + nstar][app]) * 10 ** 10

            ssfr = this_ini_ms / 0.1 / gal_m

            # Extract this galaxies information
            hmr = hmrs[ind]
            ghmr = ghmrs[ind]
            g, sg = grps[ind], subgrps[ind]

            # Whats the MEGA ID of this galaxy?
            mega_ind = np.where(np.logical_and(mega_grps == g,
                                               mega_subgrps == sg))[0]

            # Get the progenitor
            start = start_index[mega_ind][0]
            stride = nprogs[mega_ind][0]
            main_prog = prog_ids[start]

            if stride == 0:
                no_prog_ssfr.append(ssfr)
                no_prog_mass.append(gal_m)
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
            prog_ghmr = prog_ghmrs[flares_ind]

            if prog_hmr.size == 0:
                no_prog_ssfr.append(ssfr)
                no_prog_mass.append(gal_m)
                continue

            # Include these results for plotting
            tot_hmrs.append(hmr)
            tot_prog_hmrs.extend(prog_hmr)
            tot_hmrs_gas.append(ghmr)
            tot_prog_hmrs_gas.extend(prog_ghmr)
            tot_mass.append(gal_m)
            tot_ssfr.append(ssfr)
            w.append(ws[ind])

    # Convert to arrays
    tot_hmrs = np.array(tot_hmrs)
    tot_prog_hmrs = np.array(tot_prog_hmrs)
    tot_hmrs_gas = np.array(tot_hmrs_gas)
    tot_prog_hmrs_gas = np.array(tot_prog_hmrs_gas)
    tot_ssfr = np.array(tot_ssfr)
    tot_mass = np.array(tot_mass)
    w = np.array(w)
    if len(no_prog_ssfr) > 0:
        no_prog_ssfr = np.array(no_prog_ssfr)
        no_prog_mass = np.array(no_prog_mass)

    # Define delta
    delta_hmr = tot_hmrs / tot_prog_hmrs
    delta_ghmr = tot_hmrs_gas / tot_prog_hmrs_gas

    # Set up plot
    fig = plt.figure(figsize=(3 * 3.5 + 0.15, 3.5))
    gs = gridspec.GridSpec(1, 4, width_ratios=[20, 20, 20, 1])
    gs.update(wspace=0.0, hspace=0.0)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    cax = fig.add_subplot(gs[0, 3])
    ax1.loglog()
    ax2.loglog()
    ax3.loglog()

    okinds = np.logical_and(tot_ssfr > 0, np.logical_and(delta_hmr > 0,
                                                         delta_ghmr > 0))
    delta_hmr = delta_hmr[okinds]
    delta_ghmr = delta_ghmr[okinds]
    tot_ssfr = tot_ssfr[okinds]
    tot_hmrs = tot_hmrs[okinds]
    tot_prog_hmrs = tot_prog_hmrs[okinds]
    w = w[okinds]

    # Plot the data
    okinds = np.logical_and(tot_hmrs > 1, tot_prog_hmrs > 1)
    im = ax1.hexbin(delta_hmr[okinds], tot_ssfr[okinds],  gridsize=50,
                    mincnt=np.min(w) - (0.1 * np.min(w)),
                    C=w[okinds], xscale="log", yscale="log",
                    reduce_C_function=np.mean, norm=weight_norm,
                    linewidths=0.2, cmap="plasma")
    ax1.set_title("$R_{1/2,\star}^{A} > 1 && R_{1/2,\star}^{B} > 1$")
    okinds = np.logical_and(tot_hmrs <= 1, tot_prog_hmrs > 1)
    im = ax2.hexbin(delta_hmr[okinds], tot_ssfr[okinds],  gridsize=50,
                    mincnt=np.min(w) - (0.1 * np.min(w)),
                    C=w[okinds], xscale="log", yscale="log",
                    reduce_C_function=np.mean, norm=weight_norm,
                    linewidths=0.2, cmap="plasma")
    ax2.set_title("$R_{1/2,\star}^{A} \leq 1 && R_{1/2,\star}^{B} > 1$")
    okinds = np.logical_and(tot_hmrs <= 1, tot_prog_hmrs <= 1)
    im = ax3.hexbin(delta_hmr[okinds], tot_ssfr[okinds],  gridsize=50,
                    mincnt=np.min(w) - (0.1 * np.min(w)),
                    C=w[okinds], xscale="log", yscale="log",
                    reduce_C_function=np.mean, norm=weight_norm,
                    linewidths=0.2, cmap="plasma")
    ax3.set_title("$R_{1/2,\star}^{A} \leq 1 && R_{1/2,\star}^{B} \leq 1$")

    # Axes labels
    ax1.set_xlabel("$R_{1/2, \star}^{B} / R_{1/2, \star}^{A}$")
    ax2.set_xlabel("$R_{1/2, \star}^{B} / R_{1/2, \star}^{A}$")
    ax3.set_xlabel("$R_{1/2, \star}^{B} / R_{1/2, \star}^{A}$")
    ax1.set_ylabel("$\mathrm{sSFR} / [\mathrm{Gyr}^{-1}]$")

    cbar = fig.colorbar(im, cax)
    cbar.set_label("$\sum w_i$")

    # Get and set universal axis limits
    xmin, xmax = np.inf, 0
    ymin, ymax = np.inf, 0
    for ax in [ax1, ax2, ax3]:
        xlims = ax.get_xlim()
        ylims = ax.get_ylim()
        if xlims[0] < xmin:
            xmin = xlims[0]
        if ylims[0] < ymin:
            ymin = ylims[0]
        if xlims[1] > xmax:
            xmax = xlims[1]
        if ylims[1] > ymax:
            ymax = ylims[1]
    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

    # Save figure
    mkdir("plots/graph/")
    fig.savefig("plots/graph/delta_hmr_ssfr_mass.png",
                bbox_inches="tight")
    plt.close(fig)

    # Set up plot
    fig = plt.figure(figsize=(3 * 3.5 + 0.15, 3.5))
    gs = gridspec.GridSpec(1, 4, width_ratios=[20, 20, 20, 1])
    gs.update(wspace=0.0, hspace=0.0)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    cax = fig.add_subplot(gs[0, 3])
    ax1.loglog()
    ax2.loglog()
    ax3.loglog()

    # Plot the data
    okinds = np.logical_and(tot_hmrs > 1, tot_prog_hmrs > 1)
    im = ax1.hexbin(delta_ghmr[okinds], delta_hmr[okinds],  gridsize=50,
                    mincnt=np.min(w) - (0.1 * np.min(w)),
                    C=w[okinds], xscale="log", yscale="log",
                    reduce_C_function=np.mean, norm=weight_norm,
                    linewidths=0.2, cmap="plasma")
    ax1.set_title("$R_{1/2,\star}^{A} > 1 && R_{1/2,\star}^{B} > 1$")
    okinds = np.logical_and(tot_hmrs <= 1, tot_prog_hmrs > 1)
    im = ax2.hexbin(delta_ghmr[okinds], delta_hmr[okinds],  gridsize=50,
                    mincnt=np.min(w) - (0.1 * np.min(w)),
                    C=w[okinds], xscale="log", yscale="log",
                    reduce_C_function=np.mean, norm=weight_norm,
                    linewidths=0.2, cmap="plasma")
    ax2.set_title("$R_{1/2,\star}^{A} \leq 1 && R_{1/2,\star}^{B} > 1$")
    okinds = np.logical_and(tot_hmrs <= 1, tot_prog_hmrs <= 1)
    im = ax3.hexbin(delta_ghmr[okinds], delta_hmr[okinds],  gridsize=50,
                    mincnt=np.min(w) - (0.1 * np.min(w)),
                    C=w[okinds], xscale="log", yscale="log",
                    reduce_C_function=np.mean, norm=weight_norm,
                    linewidths=0.2, cmap="plasma")
    ax3.set_title("$R_{1/2,\star}^{A} \leq 1 && R_{1/2,\star}^{B} \leq 1$")

    # Axes labels
    ax1.set_xlabel("$R_{1/2, \mathrm{gas}}^{B} / R_{1/2, \mathrm{gas}}^{A}$")
    ax2.set_xlabel("$R_{1/2, \mathrm{gas}}^{B} / R_{1/2, \mathrm{gas}}^{A}$")
    ax3.set_xlabel("$R_{1/2, \mathrm{gas}}^{B} / R_{1/2, \mathrm{gas}}^{A}$")
    ax.set_ylabel("$R_{1/2, \star}^{B} / R_{1/2, \star}^{A}$")

    # Get and set universal axis limits
    xmin, xmax = np.inf, 0
    ymin, ymax = np.inf, 0
    for ax in [ax1, ax2, ax3]:
        xlims = ax.get_xlim()
        ylims = ax.get_ylim()
        if xlims[0] < xmin:
            xmin = xlims[0]
        if ylims[0] < ymin:
            ymin = ylims[0]
        if xlims[1] > xmax:
            xmax = xlims[1]
        if ylims[1] > ymax:
            ymax = ylims[1]
    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

    cbar = fig.colorbar(im, cax)
    cbar.set_label("$\sum w_i$")

    # Save figure
    mkdir("plots/graph/")
    fig.savefig("plots/graph/gas_delta_hmr_stellar_delta_hmr.png",
                bbox_inches="tight")
    plt.close(fig)

    # Set up plot
    fig = plt.figure(figsize=(3.5, 3.5))
    ax = fig.add_subplot(111)
    ax.loglog()

    okinds = np.logical_and(tot_mass > 0, tot_ssfr > 0)
    okinds = np.logical_and(okinds, tot_prog_hmrs > 1)

    # Plot the scatter
    im = ax.hexbin(delta_hmr[okinds], tot_ssfr[okinds],  gridsize=50,
                   mincnt=np.min(w) - (0.1 * np.min(w)),
                   C=w[okinds], xscale="log", yscale="log",
                   reduce_C_function=np.sum, norm=weight_norm,
                   linewidths=0.2, cmap="plasma")

    # Axes labels
    ax.set_xlabel("$R_{1/2}^{B} / R_{1/2}^{A}$")
    ax.set_ylabel("$\mathrm{sSFR} / [\mathrm{Gyr}^{-1}]$")

    cbar = fig.colorbar(im)
    cbar.set_label("$\sum w_i$")

    # Save figure
    mkdir("plots/graph/")
    fig.savefig("plots/graph/delta_hmr_ssfr_mass_subset.png",
                bbox_inches="tight")
    plt.close(fig)


def plot_size_feedback(stellar_data, other_data, snaps, weight_norm, plt_type):

    # Get the dark matter mass
    hdf = h5py.File("/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/flares_00/"
                    "data/snapshot_000_z015p000/snap_000_z015p000.0.hdf5")
    mdm = hdf["Header"].attrs["MassTable"][1]
    hdf.close()

    # Define paths
    path = "/cosma/home/dp004/dc-rope1/cosma7/FLARES/flares-mergergraph/"
    halo_base = path + "data/halos/MEGAFLARES_halos_<reg>_<snap>.hdf5"
    graph_base = path + "data/dgraph/MEGAFLARES_graph_<reg>_<snap>.hdf5"
    master_base = "/cosma7/data/dp004/dc-payy1/my_files/flares_pipeline/data/flares.hdf5"

    # Split snapshots into current and progenitor lists
    current_snaps = snaps[1:]
    prog_snaps = snaps[:-1]

    # Initialise lists for storing results
    tot_hmrs = []
    tot_prog_hmrs = []
    binding_energy = []
    feedback_energy = []
    prog_binding_energy = []
    prog_feedback_energy = []
    w = []

    # Open the master file
    hdf_master = h5py.File(master_base, "r")

    # Physical constants
    G = (const.G.to(u.Mpc ** 3 * u.M_sun ** -1 * u.yr ** -2)).value

    # Loop over snapshots
    for snap, prog_snap in zip(current_snaps, prog_snaps):

        if snap != current_snaps[-1]:
            continue

        print(snap, prog_snap)

        # Get redshift
        z = float(snap.split("z")[-1].replace("p", "."))
        prog_z = float(prog_snap.split("z")[-1].replace("p", "."))

        # Set fake region IDs
        reg = "100"
        reg_int = -1

        # Extract galaxy data from the sizes dict
        hmrs = stellar_data[snap]["HMRs"][...]
        cuton_hmrs = other_data[snap]["HMRs"][...]
        print("There are %d galaxies" % len(hmrs))
        print("There are %d compact galaxies" % len(hmrs[hmrs < 1]))
        prog_hmrs = stellar_data[prog_snap]["HMRs"][...]
        grps = stellar_data[snap]["Galaxy,GroupNumber"][...]
        subgrps = stellar_data[snap]["Galaxy,SubGroupNumber"][...]
        prog_grps = stellar_data[prog_snap]["Galaxy,GroupNumber"][...]
        prog_subgrps = stellar_data[prog_snap]["Galaxy,SubGroupNumber"][...]
        regions = stellar_data[snap]["regions"][...]
        ws = stellar_data[snap]["weights"][...]
        prog_regions = stellar_data[prog_snap]["regions"][...]

        # Loop over galaxies
        for ind in range(len(hmrs)):

            # Skip this galaxy if it is not compact
            if cuton_hmrs[ind] > 1:
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
                prog_ids = hdf_graph["ProgHaloIDs"][...]
                start_index = hdf_graph["prog_start_index"][...]

                hdf_halo.close()
                hdf_prog.close()
                hdf_graph.close()

                # Get the master file group
                snap_grp = hdf_master[reg][snap]
                prog_grp = hdf_master[reg][prog_snap]

                # Get the group and subgroup id arrays from the master file
                master_grps = snap_grp["Galaxy"]["GroupNumber"][...]
                master_subgrps = snap_grp["Galaxy"]["SubGroupNumber"][...]
                prog_master_grps = prog_grp["Galaxy"]["GroupNumber"][...]
                prog_master_subgrps = prog_grp["Galaxy"]["SubGroupNumber"][...]

                # Get other data from the master file
                master_s_length = snap_grp["Galaxy"]["S_Length"][...]
                ini_ms = snap_grp["Particle"]["S_MassInitial"][...]
                prog_master_s_length = prog_grp["Galaxy"]["S_Length"][...]
                prog_ini_ms = prog_grp["Particle"]["S_MassInitial"][...]

            # Extract this galaxies information
            hmr = hmrs[ind]
            g, sg = grps[ind], subgrps[ind]

            # Whats the MEGA ID of this galaxy?
            mega_ind = np.where(np.logical_and(mega_grps == g,
                                               mega_subgrps == sg))[0]

            # Get the progenitor
            start = start_index[mega_ind][0]
            main_prog = prog_ids[start]

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

            # Get the index in the master file
            master_ind = np.where(
                np.logical_and(master_grps == g,
                               master_subgrps == sg)
            )[0]
            prog_master_ind = np.where(
                np.logical_and(prog_master_grps == prog_g,
                               prog_master_subgrps == prog_sg)
            )[0]

            if master_ind.size == 0:
                continue
            if prog_master_ind.size == 0:
                continue

            # Extract the index from the array it's contained within
            master_ind = master_ind[0]
            prog_master_ind = prog_master_ind[0]

            # Get the start index for each particle type
            s_start = np.sum(master_s_length[:master_ind])
            s_len = master_s_length[master_ind]

            prog_s_start = np.sum(prog_master_s_length[:prog_master_ind])
            prog_s_len = prog_master_s_length[prog_master_ind]

            # Get this galaxy's data
            this_ini_ms = ini_ms[s_start: s_start + s_len]
            prog_this_ini_ms = prog_ini_ms[
                prog_s_start: prog_s_start + prog_s_len]

            # Include these results for plotting
            tot_hmrs.append(hmr)
            tot_prog_hmrs.extend(prog_hmr)
            feedback_energy.append(np.sum(1.74 * 10 ** 49 * this_ini_ms))
            prog_feedback_energy.append(np.sum(1.74 * 10 ** 49 *
                                               prog_this_ini_ms))
            w.append(ws[ind])

    hdf_master.close()

    # Convert to arrays
    tot_hmrs = np.array(tot_hmrs)
    tot_prog_hmrs = np.array(tot_prog_hmrs)
    feedback_energy = np.array(feedback_energy)
    prog_feedback_energy = np.array(prog_feedback_energy)
    w = np.array(w)

    # Define deltas
    delta_hmr = tot_hmrs / tot_prog_hmrs
    delta_fb = feedback_energy / prog_feedback_energy

    # Set up plot
    fig = plt.figure(figsize=(3.5, 3.5))
    ax = fig.add_subplot(111)
    ax.loglog()

    okinds = np.logical_and(delta_fb > 0, delta_hmr > 0)

    # Plot the scatter
    im = ax.hexbin(delta_fb[okinds], delta_hmr[okinds], gridsize=50,
                   mincnt=np.min(w) - (0.1 * np.min(w)),
                   C=w[okinds], xscale="log", yscale="log",
                   reduce_C_function=np.sum, norm=weight_norm,
                   linewidths=0.2, cmap="plasma")

    # Axes labels
    ax.set_xlabel(
        "$E_{\star\mathrm{fb}}^\mathrm{B} / E_{\star\mathrm{fb}}^\mathrm{A}$")
    ax.set_ylabel("$R_{1/2}^{B} / R_{1/2}^{A}$")

    cbar = fig.colorbar(im)
    cbar.set_label("$\sum w_i$")

    # Save figure
    mkdir("plots/graph/")
    fig.savefig("plots/graph/delta_hmr_fb_%s.png" % plt_type,
                bbox_inches="tight")
    plt.close(fig)


def plot_size_mass_evo_grid_noncompact(stellar_data, snaps):

    # Define paths
    path = "/cosma/home/dp004/dc-rope1/cosma7/FLARES/flares-mergergraph/"
    halo_base = path + "data/halos/MEGAFLARES_halos_<reg>_<snap>.hdf5"
    graph_base = path + "data/dgraph/MEGAFLARES_graph_<reg>_<snap>.hdf5"

    # Set up the dictionary to store the graph information
    graph = {}

    # Define root snapshot
    root_snap = snaps[-1]

    # Extract galaxy data from the sizes dict for the root snap
    root_hmrs = stellar_data[root_snap]["HMRs"][...]
    root_mass = stellar_data[root_snap]["mass"][...]
    root_grps = stellar_data[root_snap]["Galaxy,GroupNumber"][...]
    root_subgrps = stellar_data[root_snap]["Galaxy,SubGroupNumber"][...]
    root_regions = stellar_data[root_snap]["regions"][...]

    # Define redshift norm
    norm = cm.Normalize(vmin=5, vmax=12)

    # Create data dictionary to speed up walking
    mega_data = {}
    for reg_int in np.unique(root_regions):
        if reg_int == 18:
            continue
        reg = str(reg_int).zfill(2)
        mega_data[reg] = {}
        for snap in snaps:
            mega_data[reg][snap] = {}

            # Open this new region
            this_halo_base = halo_base.replace("<reg>", reg)
            this_halo_base = this_halo_base.replace("<snap>", snap)
            this_graph_base = graph_base.replace("<reg>", reg)
            this_graph_base = this_graph_base.replace("<snap>", snap)
            hdf_halo = h5py.File(this_halo_base, "r")
            hdf_graph = h5py.File(this_graph_base, "r")

            # Get the MEGA ID arrays for both snapshots
            mega_data[reg][snap]["group_number"] = hdf_halo["group_number"][...]
            mega_data[reg][snap]["subgroup_number"] = hdf_halo["subgroup_number"][...]

            # Get the progenitor information
            mega_data[reg][snap]["ProgHaloIDs"] = hdf_graph["ProgHaloIDs"][...]
            mega_data[reg][snap]["prog_start_index"] = hdf_graph["prog_start_index"][...]

            hdf_halo.close()
            hdf_graph.close()

    # Loop over galaxies and populate the root level of the graph with
    # only the compact galaxies
    for ind in range(len(root_hmrs)):

        # Skip if the galaxy isn't compact
        if root_hmrs[ind] < 1:
            continue

        # Get ID
        g, sg = root_grps[ind], root_subgrps[ind]

        # Make an entry in the dict for it
        graph[(g, sg, ind)] = {"HMRs": [], "Masses": [], "z": []}

    # Loop over these root galaxies and populate the rest of the graph
    i = 0
    for key in graph:

        print("Walking %d (%d, %d, %d) of %d" % (i, key[0], key[1],
                                                 key[2], len(graph)), end="\r")

        # Extract IDs
        g, sg, ind = key

        # Get the region for this galaxy
        reg_int = root_regions[ind]
        if reg_int == 18:
            continue
        reg = str(reg_int).zfill(2)

        # Set up the snapshot
        snap_ind = len(snaps) - 1
        snap = root_snap
        prog_snap = snaps[snap_ind - 1]

        # Set up looping variables
        this_g, this_sg, this_ind = g, sg, ind

        i += 1

        # Loop until we don't have a progenitor to step to
        while snap_ind >= 0:

            # Get redshift
            z = float(snap.split("z")[-1].replace("p", "."))

            # Extract galaxy data from the sizes dict
            hmrs = stellar_data[snap]["HMRs"][...]
            mass = stellar_data[snap]["mass"][...]
            prog_grps = stellar_data[prog_snap]["Galaxy,GroupNumber"][...]
            prog_subgrps = stellar_data[prog_snap]["Galaxy,SubGroupNumber"][...]
            prog_regions = stellar_data[prog_snap]["regions"][...]

            # Put this galaxy in the graph
            if snap == root_snap:
                graph[(g, sg, ind)]["HMRs"].append(
                    hmrs[this_ind]  # / stellar_data[root_snap]["HMRs"][ind]
                )
                graph[(g, sg, ind)]["Masses"].append(
                    mass[this_ind]
                )
            else:
                graph[(g, sg, ind)]["HMRs"].extend(
                    hmrs[this_ind]  # / stellar_data[root_snap]["HMRs"][ind]
                )
                graph[(g, sg, ind)]["Masses"].extend(
                    mass[this_ind]
                )
            graph[(g, sg, ind)]["z"].append(z)

            # Get the MEGA ID arrays for both snapshots
            mega_grps = mega_data[reg][snap]["group_number"]
            mega_subgrps = mega_data[reg][snap]["subgroup_number"]
            mega_prog_grps = mega_data[reg][prog_snap]["group_number"]
            mega_prog_subgrps = mega_data[reg][prog_snap]["subgroup_number"]

            # Get the progenitor information
            prog_ids = mega_data[reg][snap]["ProgHaloIDs"]
            start_index = mega_data[reg][snap]["prog_start_index"]

            # Whats the MEGA ID of this galaxy?
            mega_ind = np.where(np.logical_and(mega_grps == this_g,
                                               mega_subgrps == this_sg))[0]

            # Get the progenitor
            start = start_index[mega_ind][0]
            main_prog = prog_ids[start]

            if start != 2**30:

                # Get this progenitors group and subgroup ID
                this_g = mega_prog_grps[main_prog]
                this_sg = mega_prog_subgrps[main_prog]

                # Get this progenitors index
                this_ind = np.where(
                    np.logical_and(prog_regions == reg_int,
                                   np.logical_and(prog_grps == this_g,
                                                  prog_subgrps == this_sg)))[0]

                if this_ind.size == 0:
                    break

                # Set snapshots
                snap_ind -= 1
                snap = snaps[snap_ind]
                prog_snap = snaps[snap_ind - 1]

            else:
                break

    # Set up plot parameters
    ylims = (10**-1.3, 10**1.3)
    xlims = (10**8, 10**11.5)

    # Define mins and maxs for binning
    min_hmr = np.inf
    max_hmr = 0

    # Get the max size reached in each main branch
    max_size = {}
    for key in list(graph.keys()):
        if len(graph[key]["HMRs"]) > 1:
            max_size[key] = graph[key]["HMRs"][-1]
            if max_size[key] > max_hmr:
                max_hmr = max_size[key]
            if max_size[key] < min_hmr:
                min_hmr = max_size[key]
        else:
            del graph[key]

    # Create array of max sizes to bin
    keys = list(graph.keys())
    max_size_arr = np.zeros(len(keys))
    max_mass_arr = np.zeros(len(keys))
    for ind, key in enumerate(keys):
        max_size_arr[ind] = max_size[key]
        max_mass_arr[ind] = graph[key]["Masses"][0]

    # Define size bins
    size_bin_edges = np.logspace(
        np.log10(np.min(max_size_arr) - 0.01 * np.min(max_size_arr)),
        np.log10(np.max(max_size_arr) + 0.01 * np.max(max_size_arr)),
        5)
    mass_bin_edges = [10**8, 10**8.5, 10**9, 10 **
                      9.5, 10**10, np.max(max_mass_arr) + 10**9]
    size_bins = np.digitize(max_size_arr, size_bin_edges) - 1
    mass_bins = np.digitize(max_mass_arr, mass_bin_edges) - 1

    print(np.unique(size_bins, return_counts=True))
    print(np.unique(mass_bins, return_counts=True))

    # Define plot grid shape
    nrows = len(size_bin_edges) - 1
    ncols = len(mass_bin_edges) - 1

    # Set up plot
    fig = plt.figure(figsize=(3.5 * ncols + 0.1 * 3.5, 3.5 * nrows))
    gs = gridspec.GridSpec(nrows=nrows, ncols=ncols + 1,
                           width_ratios=[20, ] * ncols + [1])
    gs.update(wspace=0.0, hspace=0.0)
    axes = np.empty((nrows, ncols), dtype=object)
    cax = fig.add_subplot(gs[:, -1])

    cax.tick_params(axis="both", top=False, bottom=False,
                    left=False, right=False,
                    labeltop=False, labelbottom=False,
                    labelleft=False, labelright=False)

    for i in range(nrows):
        for j in range(ncols):
            axes[i, j] = fig.add_subplot(gs[i, j])
            axes[i, j].loglog()
            axes[i, j].set_xlim(xlims)
            axes[i, j].set_ylim(ylims)
            if i < nrows - 1:
                axes[i, j].tick_params(axis='x', top=False, bottom=False,
                                       labeltop=False, labelbottom=False)
            if j > 0:
                axes[i, j].tick_params(axis='y', left=False, right=False,
                                       labelleft=False, labelright=False)

            if i == 0:
                axes[i, j].set_title("$%.1f \leq \log_{10}(M_\star^{z=5} / M_\odot) < %.1f$"
                                     % (np.log10(mass_bin_edges[j]),
                                        np.log10(mass_bin_edges[j + 1])))

            if j == 0:
                axes[i, j].annotate(
                    "$%.1f \leq R_{1/2}^{\mathrm{form}} / [\mathrm{pkpc}] < %.1f$"
                    % (size_bin_edges[i], size_bin_edges[i + 1]),
                    xy=(0, 0.5), xytext=(-axes[i, j].yaxis.labelpad - 5, 0),
                    xycoords=axes[i,
                                  j].yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center', rotation=90)

    # Loop over graphs
    ii = 0
    for ind, key in enumerate(keys):

        print("Plotting %d (%d, %d, %d) of %d" % (ii, key[0], key[1],
                                                  key[2], len(graph)), end="\r")

        i, j = size_bins[ind], mass_bins[ind]

        # Skip galaxies that fell outside the range
        if i == -1 or j == -1:
            continue

        # Plot the scatter
        im = axes[i, j].plot(graph[key]["Masses"], graph[key]["HMRs"],
                             color="grey", alpha=0.2, zorder=0)
        ii += 1

    # Loop over graphs
    ii = 0
    for ind, key in enumerate(keys):

        print("Plotting %d (%d, %d, %d) of %d" % (ii, key[0], key[1],
                                                  key[2], len(graph)), end="\r")

        i, j = size_bins[ind], mass_bins[ind]

        # Skip galaxies that fell outside the range
        if i == -1 or j == -1:
            continue

        # Plot the scatter
        im = axes[i, j].scatter(graph[key]["Masses"], graph[key]["HMRs"],
                                marker=".", edgecolors="none", s=20,
                                c=graph[key]["z"], cmap="cmr.chroma",
                                alpha=0.8, zorder=1, norm=norm)
        ii += 1

    # Axes labels
    for ax in axes[:, 0]:
        ax.set_ylabel("$R_{1/2\star} / [\mathrm{pkpc}]$")
    for ax in axes[-1, :]:
        ax.set_xlabel("$M_{\star} / M_{\odot}$")

    cbar = fig.colorbar(im, cax)
    cbar.set_label("$z$")

    # Save figure
    mkdir("plots/graph_plot/")
    fig.savefig("plots/graph_plot/size_mass_evo_all_noncompact.png",
                bbox_inches="tight")
    plt.close(fig)


def plot_formation_size_environ(stellar_data, snaps):

    # Open region overdensities
    reg_ovdens = np.loadtxt("/cosma7/data/dp004/dc-rope1/FLARES/"
                            "flares/region_overdensity.txt",
                            dtype=float)

    # Define paths
    path = "/cosma/home/dp004/dc-rope1/cosma7/FLARES/flares-mergergraph/"
    halo_base = path + "data/halos/MEGAFLARES_halos_<reg>_<snap>.hdf5"
    graph_base = path + "data/dgraph/MEGAFLARES_graph_<reg>_<snap>.hdf5"

    # Set up the dictionary to store the graph information
    graph = {}
    form_size = []
    form_z = []
    form_ovden = []

    # Define root snapshot
    root_snap = snaps[-1]

    # Extract galaxy data from the sizes dict for the root snap
    root_hmrs = stellar_data[root_snap]["HMRs"][...]
    root_mass = stellar_data[root_snap]["mass"][...]
    root_grps = stellar_data[root_snap]["Galaxy,GroupNumber"][...]
    root_subgrps = stellar_data[root_snap]["Galaxy,SubGroupNumber"][...]
    root_regions = stellar_data[root_snap]["regions"][...]

    # Define redshift norm
    norm = cm.Normalize(vmin=5, vmax=12)

    # Create data dictionary to speed up walking
    mega_data = {}
    for reg_int in np.unique(root_regions):
        if reg_int == 18:
            continue
        reg = str(reg_int).zfill(2)
        mega_data[reg] = {}
        for snap in snaps:
            mega_data[reg][snap] = {}

            # Open this new region
            this_halo_base = halo_base.replace("<reg>", reg)
            this_halo_base = this_halo_base.replace("<snap>", snap)
            this_graph_base = graph_base.replace("<reg>", reg)
            this_graph_base = this_graph_base.replace("<snap>", snap)
            hdf_halo = h5py.File(this_halo_base, "r")
            hdf_graph = h5py.File(this_graph_base, "r")

            # Get the MEGA ID arrays for both snapshots
            mega_data[reg][snap]["group_number"] = hdf_halo["group_number"][...]
            mega_data[reg][snap]["subgroup_number"] = hdf_halo["subgroup_number"][...]

            # Get the progenitor information
            mega_data[reg][snap]["ProgHaloIDs"] = hdf_graph["ProgHaloIDs"][...]
            mega_data[reg][snap]["prog_start_index"] = hdf_graph["prog_start_index"][...]

            hdf_halo.close()
            hdf_graph.close()

    # Loop over galaxies and populate the root level of the graph with
    # only the compact galaxies
    for ind in range(len(root_hmrs)):

        # Get ID
        g, sg = root_grps[ind], root_subgrps[ind]

        # Make an entry in the dict for it
        graph[(g, sg, ind)] = {"HMRs": [], "Masses": [], "z": []}

    # Loop over these root galaxies and populate the rest of the graph
    i = 0
    for key in graph:

        print("Walking %d (%d, %d, %d) of %d" % (i, key[0], key[1],
                                                 key[2], len(graph)), end="\r")

        # Extract IDs
        g, sg, ind = key

        # Get the region for this galaxy
        reg_int = root_regions[ind]
        if reg_int == 18:
            continue
        reg = str(reg_int).zfill(2)

        # Set up the snapshot
        snap_ind = len(snaps) - 1
        snap = root_snap
        prog_snap = snaps[snap_ind - 1]

        # Set up looping variables
        this_g, this_sg, this_ind = g, sg, ind

        i += 1

        # Loop until we don't have a progenitor to step to
        while snap_ind >= 0:

            # Get redshift
            z = float(snap.split("z")[-1].replace("p", "."))

            # Extract galaxy data from the sizes dict
            hmrs = stellar_data[snap]["HMRs"][...]
            mass = stellar_data[snap]["mass"][...]
            prog_grps = stellar_data[prog_snap]["Galaxy,GroupNumber"][...]
            prog_subgrps = stellar_data[prog_snap]["Galaxy,SubGroupNumber"][...]
            prog_regions = stellar_data[prog_snap]["regions"][...]

            # Put this galaxy in the graph
            if snap == root_snap:
                graph[(g, sg, ind)]["HMRs"].append(
                    hmrs[this_ind]  # / stellar_data[root_snap]["HMRs"][ind]
                )
                graph[(g, sg, ind)]["Masses"].append(
                    mass[this_ind]
                )
            else:
                graph[(g, sg, ind)]["HMRs"].extend(
                    hmrs[this_ind]  # / stellar_data[root_snap]["HMRs"][ind]
                )
                graph[(g, sg, ind)]["Masses"].extend(
                    mass[this_ind]
                )
            graph[(g, sg, ind)]["z"].append(z)

            # Get the MEGA ID arrays for both snapshots
            mega_grps = mega_data[reg][snap]["group_number"]
            mega_subgrps = mega_data[reg][snap]["subgroup_number"]
            mega_prog_grps = mega_data[reg][prog_snap]["group_number"]
            mega_prog_subgrps = mega_data[reg][prog_snap]["subgroup_number"]

            # Get the progenitor information
            prog_ids = mega_data[reg][snap]["ProgHaloIDs"]
            start_index = mega_data[reg][snap]["prog_start_index"]

            # Whats the MEGA ID of this galaxy?
            mega_ind = np.where(np.logical_and(mega_grps == this_g,
                                               mega_subgrps == this_sg))[0]

            # Get the progenitor
            start = start_index[mega_ind][0]
            main_prog = prog_ids[start]

            if start != 2**30:

                # Get this progenitors group and subgroup ID
                this_g = mega_prog_grps[main_prog]
                this_sg = mega_prog_subgrps[main_prog]

                # Get this progenitors index
                this_ind = np.where(
                    np.logical_and(prog_regions == reg_int,
                                   np.logical_and(prog_grps == this_g,
                                                  prog_subgrps == this_sg)))[0]

                if this_ind.size == 0:
                    break

                # Set snapshots
                snap_ind -= 1
                snap = snaps[snap_ind]
                prog_snap = snaps[snap_ind - 1]

            else:
                break

        # Store results
        form_z.append(graph[(g, sg, ind)]["z"][-1])
        form_size.append(graph[(g, sg, ind)]["HMRs"][-1])
        form_ovden.append(reg_ovdens[reg_int])

    # Set up plot
    fig = plt.figure(figsize=(3.5, 3.5))
    ax = fig.add_subplot(111)
    ax.semilogy()

    im = ax.hexbin(form_z, form_size, gridsize=50,
                   mincnt=np.min(form_ovden) - (0.1 * np.min(form_ovden)),
                   C=form_ovden, yscale="log",
                   reduce_C_function=np.mean,
                   linewidths=0.2, cmap="plasma")

    ax.set_xlabel("$z_\mathrm{form}$")
    ax.set_ylabel("$R_{1/2}^\mathrm{form} / [\mathrm{pkpc}]$")

    cbar = fig.colorbar(im)
    cbar.set_label("$\log_{10}(1+\Delta)$")

    # Save figure
    mkdir("plots/graph_plot/")
    fig.savefig("plots/graph_plot/formation_environment_size.png",
                bbox_inches="tight")
    plt.close(fig)


def plot_size_change_starpos(stellar_data, snaps, weight_norm):

    # Get the dark matter mass
    hdf = h5py.File("/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/flares_00/"
                    "data/snapshot_000_z015p000/snap_000_z015p000.0.hdf5")
    mdm = hdf["Header"].attrs["MassTable"][1]
    hdf.close()

    # Define paths
    path = "/cosma/home/dp004/dc-rope1/cosma7/FLARES/flares-mergergraph/"
    halo_base = path + "data/halos/MEGAFLARES_halos_<reg>_<snap>.hdf5"
    graph_base = path + "data/dgraph/MEGAFLARES_graph_<reg>_<snap>.hdf5"
    master_base = "/cosma7/data/dp004/dc-payy1/my_files/flares_pipeline/data/flares.hdf5"

    # Split snapshots into current and progenitor lists
    current_snaps = snaps[1:]
    prog_snaps = snaps[:-1]

    # Initialise lists for storing results
    tot_hmrs = []
    tot_prog_hmrs = []
    tot_prog_rs = []
    tot_rs = []
    tot_ssfrs = []
    w = []
    tot_cops = []
    prog_tot_cops = []
    tot_mass = []

    # Open the master file
    hdf_master = h5py.File(master_base, "r")

    # Loop over snapshots
    for snap, prog_snap in zip(current_snaps, prog_snaps):

        if snap != current_snaps[-1]:
            continue

        print(snap, prog_snap)

        # Get redshift
        z = float(snap.split("z")[-1].replace("p", "."))
        prog_z = float(prog_snap.split("z")[-1].replace("p", "."))

        # Define softening length
        soft = 0.001802390 / (0.6777 * (1 + z))
        prog_soft = 0.001802390 / (0.6777 * (1 + prog_z))

        # Set fake region IDs
        reg = "100"
        reg_int = -1

        # Extract galaxy data from the sizes dict
        hmrs = stellar_data[snap]["HMRs"][...]
        prog_hmrs = stellar_data[prog_snap]["HMRs"][...]
        grps = stellar_data[snap]["Galaxy,GroupNumber"][...]
        subgrps = stellar_data[snap]["Galaxy,SubGroupNumber"][...]
        prog_grps = stellar_data[prog_snap]["Galaxy,GroupNumber"][...]
        prog_subgrps = stellar_data[prog_snap]["Galaxy,SubGroupNumber"][...]
        regions = stellar_data[snap]["regions"][...]
        ws = stellar_data[snap]["weights"][...]
        prog_regions = stellar_data[prog_snap]["regions"][...]

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

                # Get the master file group
                snap_grp = hdf_master[reg][snap]
                prog_grp = hdf_master[reg][prog_snap]

                # Get the group and subgroup id arrays from the master file
                master_grps = snap_grp["Galaxy"]["GroupNumber"][...]
                master_subgrps = snap_grp["Galaxy"]["SubGroupNumber"][...]
                prog_master_grps = prog_grp["Galaxy"]["GroupNumber"][...]
                prog_master_subgrps = prog_grp["Galaxy"]["SubGroupNumber"][...]

                # Get other data from the master file
                cops = snap_grp["Galaxy"]["COP"][...].T
                master_s_length = snap_grp["Galaxy"]["S_Length"][...]
                master_s_age = snap_grp["Particle"]["S_Age"][...]
                master_s_inim = snap_grp["Particle"]["S_MassInitial"][...] * 10 ** 10
                master_app = snap_grp["Particle"]["Apertures/Star/30"][...]
                master_s_mass = snap_grp["Galaxy"]["Mstar_aperture"]["30"][...] * 10 ** 10
                master_s_pos = snap_grp["Particle"]["S_Coordinates"][...].T
                master_s_pids = snap_grp["Particle"]["S_Index"][...]
                prog_cops = prog_grp["Galaxy"]["COP"][...].T
                prog_master_s_length = prog_grp["Galaxy"]["S_Length"][...]
                prog_master_s_pos = prog_grp["Particle"]["S_Coordinates"][...].T
                prog_master_s_pids = prog_grp["Particle"]["S_Index"][...]

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

            # Get the index in the master file
            master_ind = np.where(
                np.logical_and(master_grps == g,
                               master_subgrps == sg)
            )[0]
            prog_master_ind = np.where(
                np.logical_and(prog_master_grps == prog_g,
                               prog_master_subgrps == prog_sg)
            )[0]

            if master_ind.size == 0:
                continue
            if prog_master_ind.size == 0:
                continue

            # Extract the index from the array it's contained within
            master_ind = master_ind[0]
            prog_master_ind = prog_master_ind[0]

            # Get the start index for each particle type
            s_start = np.sum(master_s_length[:master_ind])
            s_len = master_s_length[master_ind]
            cop = cops[master_ind]
            s_m = master_s_mass[master_ind]

            prog_s_start = np.sum(prog_master_s_length[:prog_master_ind])
            prog_s_len = prog_master_s_length[prog_master_ind]
            prog_cop = prog_cops[prog_master_ind]

            # Get prog data
            prog_coords = prog_master_s_pos[
                prog_s_start: prog_s_start + prog_s_len, :]
            prog_s_pids = prog_master_s_pids[
                prog_s_start: prog_s_start + prog_s_len]

            # Calculate radius and apply a 30 pkpc aperture
            prog_rs = np.sqrt((prog_coords[:, 0] - prog_cop[0]) ** 2
                              + (prog_coords[:, 1] - prog_cop[1]) ** 2
                              + (prog_coords[:, 2] - prog_cop[2]) ** 2)
            okinds = prog_rs < 0.03
            prog_rs = prog_rs[okinds]
            prog_s_pids = prog_s_pids[okinds]

            # Get this galaxy's data
            coords = master_s_pos[s_start: s_start + s_len, :]
            s_pids = master_s_pids[s_start: s_start + s_len]
            app = master_app[s_start: s_start + s_len]
            s_age = master_s_age[s_start: s_start + s_len][app]
            s_inims = master_s_inim[s_start: s_start + s_len][app]

            # Calculate sSFR
            ssfr = np.sum(s_inims[s_age < 0.1]) / 0.1 / s_m

            # Store cops
            tot_cops.append(cop)
            prog_tot_cops.append(prog_cop)
            tot_mass.append(s_m)

            # Calculate radius and apply a 30 pkpc aperture
            rs = np.sqrt((coords[:, 0] - cop[0]) ** 2
                         + (coords[:, 1] - cop[1]) ** 2
                         + (coords[:, 2] - cop[2]) ** 2)
            okinds = rs < 0.03
            rs = rs[okinds]
            s_pids = s_pids[okinds]

            # Get the particles present in the previous snapshot
            common, prog_pinds, pinds = np.intersect1d(prog_s_pids, s_pids,
                                                       return_indices=True)

            if len(common) == 0:
                continue

            # Calculate radius and apply a 30 pkpc aperture
            rs = np.sqrt((coords[pinds, 0] - cop[0]) ** 2
                         + (coords[pinds, 1] - cop[1]) ** 2
                         + (coords[pinds, 2] - cop[2]) ** 2)

            # Sort the radii
            sinds = np.argsort(s_pids[pinds])
            rs = rs[sinds]
            prog_sinds = np.argsort(prog_s_pids[prog_pinds])
            prog_rs = prog_rs[prog_pinds][prog_sinds]

            # # Include these results for plotting
            # tot_hmrs.extend(np.full(len(rs), hmr))
            # tot_prog_hmrs.extend(np.full(len(rs), prog_hmr))
            # tot_prog_rs.extend(prog_rs)
            # tot_rs.extend(rs)
            # tot_ssfrs.extend(np.full(len(rs), ssfr))
            # w.extend(np.full(len(rs), ws[ind]))

            # Include these results for plotting
            tot_hmrs.append(hmr)
            tot_prog_hmrs.extend(prog_hmr)
            tot_prog_rs.append(np.mean(prog_rs))
            tot_rs.append(np.mean(rs))
            tot_ssfrs.append(ssfr)
            w.append(ws[ind])

    hdf_master.close()

    # Convert to arrays
    tot_hmrs = np.array(tot_hmrs)
    tot_prog_hmrs = np.array(tot_prog_hmrs)
    tot_rs = np.array(tot_rs)
    prog_tot_rs = np.array(tot_prog_rs)
    tot_ssfrs = np.array(tot_ssfrs)
    w = np.array(w)
    tot_cops = np.array(tot_cops)
    prog_tot_cops = np.array(prog_tot_cops)
    tot_mass = np.array(tot_mass)

    # Compute delta
    delta_hmr = tot_hmrs / tot_prog_hmrs
    delta_rs = tot_rs / prog_tot_rs
    relative_rs = tot_rs / tot_hmrs
    delta_cop = np.sqrt((tot_cops[:, 0] - prog_tot_cops[:, 0]) ** 2
                        + (tot_cops[:, 1] - prog_tot_cops[:, 1]) ** 2
                        + (tot_cops[:, 2] - prog_tot_cops[:, 2]) ** 2)
    print(delta_hmr.size, delta_rs.size)
    print(delta_hmr)

    # Sort by decreasing size to overlay shrinking galaxies
    sinds = np.argsort(tot_ssfrs)[::-1]
    delta_hmr = delta_hmr[sinds]
    delta_rs = delta_rs[sinds]
    relative_rs = relative_rs[sinds]
    tot_ssfrs = tot_ssfrs[sinds]
    tot_rs = tot_rs[sinds]

    # Set up plot
    fig = plt.figure(figsize=(3.5, 3.5))
    ax = fig.add_subplot(111)
    ax.loglog()

    okinds = np.logical_and(delta_hmr > 0, delta_rs > 0)

    # Plot the scatter
    im = ax.scatter(delta_hmr, delta_rs, c=tot_ssfrs,
                    cmap="plasma", marker=".", norm=cm.LogNorm())

    # Axes labels
    ax.set_xlabel("$R_\mathrm{1/2}^\mathrm{B} / R_\mathrm{1/2}^\mathrm{A}$")
    ax.set_ylabel("$R_\star^\mathrm{B} / R_\star^\mathrm{A}$")

    cbar = fig.colorbar(im)
    cbar.set_label("$\mathrm{sSFR} / [\mathrm{Gyr}^{-1}]$")

    # Save figure
    mkdir("plots/graph/")
    fig.savefig("plots/graph/delta_rs_hmr.png",
                bbox_inches="tight")
    plt.close(fig)

    # Set up plot
    fig = plt.figure(figsize=(3.5, 3.5))
    ax = fig.add_subplot(111)
    ax.loglog()

    okinds = np.logical_and(relative_rs > 0, delta_rs > 0)

    # Plot the scatter
    im = ax.scatter(relative_rs, delta_rs, c=tot_ssfrs,
                    cmap="plasma", marker=".", norm=cm.LogNorm())

    # Axes labels
    ax.set_xlabel("$R_\star^\mathrm{B} / R_\mathrm{1/2}^\mathrm{B}$")
    ax.set_ylabel("$R_\star^\mathrm{B} / R_\star^\mathrm{A}$")

    cbar = fig.colorbar(im)
    cbar.set_label("$\mathrm{sSFR} / [\mathrm{Gyr}^{-1}]$")

    # Save figure
    mkdir("plots/graph/")
    fig.savefig("plots/graph/delta_rs_relR.png",
                bbox_inches="tight")
    plt.close(fig)

    # Set up plot
    fig = plt.figure(figsize=(3.5, 3.5))
    ax = fig.add_subplot(111)
    ax.loglog()

    okinds = np.logical_and(tot_rs > 0, delta_rs > 0)

    # Plot the scatter
    im = ax.scatter(tot_rs, delta_rs, c=tot_ssfrs,
                    cmap="plasma", marker=".", norm=cm.LogNorm())

    # Axes labels
    ax.set_xlabel("$R_\star^\mathrm{B} / [\mathrm{ckpc}]$")
    ax.set_ylabel("$R_\star^\mathrm{B} / R_\star^\mathrm{A}$")

    cbar = fig.colorbar(im)
    cbar.set_label("$\mathrm{sSFR} / [\mathrm{Gyr}^{-1}]$")

    # Save figure
    mkdir("plots/graph/")
    fig.savefig("plots/graph/delta_rs_R.png",
                bbox_inches="tight")
    plt.close(fig)

    # Set up plot
    fig = plt.figure(figsize=(3.5, 3.5))
    ax = fig.add_subplot(111)
    ax.loglog()

    okinds = np.logical_and(tot_rs > 0, delta_rs > 0)

    # Plot the scatter
    im = ax.scatter(delta_cop, tot_mass,
                    marker=".")

    # Axes labels
    ax.set_xlabel("$\Delta \mathrm{COP}$")
    ax.set_ylabel("$R_\star^\mathrm{B} / R_\star^\mathrm{A}$")

    cbar = fig.colorbar(im)
    cbar.set_label("$\mathrm{sSFR} / [\mathrm{Gyr}^{-1}]$")

    # Save figure
    mkdir("plots/graph/")
    fig.savefig("plots/graph/delta_rs_COP.png",
                bbox_inches="tight")
    plt.close(fig)


def plot_size_change_blackhole(stellar_data, snaps, weight_norm):

    # Define paths
    path = "/cosma/home/dp004/dc-rope1/cosma7/FLARES/flares-mergergraph/"
    halo_base = path + "data/halos/MEGAFLARES_halos_<reg>_<snap>.hdf5"
    graph_base = path + "data/dgraph/MEGAFLARES_graph_<reg>_<snap>.hdf5"
    master_base = "/cosma7/data/dp004/dc-payy1/my_files/flares_pipeline/data/flares.hdf5"

    # Split snapshots into current and progenitor lists
    current_snaps = snaps[1:]
    prog_snaps = snaps[:-1]

    # Initialise lists for storing results
    tot_hmrs = []
    tot_prog_hmrs = []
    bh_ms = []
    prog_bh_ms = []
    w = []

    # Open the master file
    hdf_master = h5py.File(master_base, "r")

    # Physical constants
    G = (const.G.to(u.Mpc ** 3 * u.M_sun ** -1 * u.Myr ** -2)).value

    # Loop over snapshots
    for snap, prog_snap in zip(current_snaps, prog_snaps):

        if snap != current_snaps[-1]:
            continue

        # Set fake region IDs
        reg = "100"
        reg_int = -1

        if snap != snaps[-1]:
            continue

        # Extract galaxy data from the sizes dict
        hmrs = stellar_data[snap]["HMRs"][...]
        prog_hmrs = stellar_data[prog_snap]["HMRs"][...]
        grps = stellar_data[snap]["Galaxy,GroupNumber"][...]
        subgrps = stellar_data[snap]["Galaxy,SubGroupNumber"][...]
        prog_grps = stellar_data[prog_snap]["Galaxy,GroupNumber"][...]
        prog_subgrps = stellar_data[prog_snap]["Galaxy,SubGroupNumber"][...]
        regions = stellar_data[snap]["regions"][...]
        ws = stellar_data[snap]["weights"][...]
        prog_regions = stellar_data[prog_snap]["regions"][...]

        # Loop over galaxies
        for ind in range(len(hmrs)):

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
                mega_prog_grps = hdf_prog["group_number"][...]
                mega_prog_subgrps = hdf_prog["subgroup_number"][...]

                # Get the progenitor information
                prog_ids = hdf_graph["ProgHaloIDs"][...]
                start_index = hdf_graph["prog_start_index"][...]
                pmasses = hdf_graph["halo_mass"][...]
                nprogs = hdf_graph["n_progs"][...]

                hdf_halo.close()
                hdf_prog.close()
                hdf_graph.close()

                # Get the master file group
                snap_grp = hdf_master[reg][snap]
                prog_grp = hdf_master[reg][prog_snap]

                # Get the group and subgroup id arrays from the master file
                master_grps = snap_grp["Galaxy"]["GroupNumber"][...]
                master_subgrps = snap_grp["Galaxy"]["SubGroupNumber"][...]
                prog_master_grps = prog_grp["Galaxy"]["GroupNumber"][...]
                prog_master_subgrps = prog_grp["Galaxy"]["SubGroupNumber"][...]

                # Get other data from the master file
                master_bh_length = snap_grp["Galaxy"]["BH_Length"][...]
                bh_mass = snap_grp["Particle"]["BH_Mass"][...] * 10 ** 10
                prog_master_bh_length = prog_grp["Galaxy"]["BH_Length"][...]
                prog_bh_mass = prog_grp["Particle"]["BH_Mass"][...] * 10 ** 10

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

            # Get the index in the master file
            master_ind = np.where(
                np.logical_and(master_grps == g,
                               master_subgrps == sg)
            )[0]
            prog_master_ind = np.where(
                np.logical_and(prog_master_grps == prog_g,
                               prog_master_subgrps == prog_sg)
            )[0]

            if master_ind.size == 0:
                continue
            if prog_master_ind.size == 0:
                continue

            # Extract the index from the array it's contained within
            master_ind = master_ind[0]
            prog_master_ind = prog_master_ind[0]

            # Get the start index for each particle type
            bh_start = np.sum(master_bh_length[:master_ind])
            bh_len = master_bh_length[master_ind]
            prog_bh_start = np.sum(prog_master_bh_length[:prog_master_ind])
            prog_bh_len = prog_master_bh_length[prog_master_ind]

            # Get this galaxy's black hole mass and it's progenitors
            this_bh_mass = bh_mass[bh_start: bh_start + bh_len]
            prog_this_bh_mass = prog_bh_mass[
                prog_bh_start: prog_bh_start + prog_bh_len]

            # Include these results for plotting
            tot_hmrs.append(hmr)
            tot_prog_hmrs.extend(prog_hmr)
            bh_ms.append(np.sum(this_bh_mass))
            prog_bh_ms.append(np.sum(prog_this_bh_mass))
            w.append(ws[ind])

    hdf_master.close()

    # Convert to arrays
    tot_hmrs = np.array(tot_hmrs)
    tot_prog_hmrs = np.array(tot_prog_hmrs)
    bh_ms = np.array(bh_ms)
    prog_bh_ms = np.array(prog_bh_ms)
    w = np.array(w)

    # Compute delta
    delta_hmr = tot_hmrs / tot_prog_hmrs
    delta_bhms = bh_ms / prog_bh_ms

    # Set up plot
    fig = plt.figure(figsize=(3 * 3.5 + 0.15, 3.5))
    gs = gridspec.GridSpec(1, 4, width_ratios=[20, 20, 20, 1])
    gs.update(wspace=0.0, hspace=0.0)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    cax = fig.add_subplot(gs[0, 3])
    ax1.loglog()
    ax2.loglog()
    ax3.loglog()

    ax2.tick_params(axis='y', left=False, right=False,
                    labelleft=False, labelright=False)
    ax3.tick_params(axis='y', left=False, right=False,
                    labelleft=False, labelright=False)

    okinds = np.logical_and(delta_bhms > 0, delta_hmr > 0)
    delta_hmr = delta_hmr[okinds]
    delta_bhms = delta_bhms[okinds]
    tot_hmrs = tot_hmrs[okinds]
    tot_prog_hmrs = tot_prog_hmrs[okinds]
    w = w[okinds]

    # Plot the data
    okinds = np.logical_and(tot_hmrs > 1, tot_prog_hmrs > 1)
    im = ax1.hexbin(delta_hmr[okinds], delta_bhms[okinds],  gridsize=30,
                    mincnt=np.min(w) - (0.1 * np.min(w)),
                    C=w[okinds], xscale="log", yscale="log",
                    reduce_C_function=np.mean, norm=weight_norm,
                    linewidths=0.2, cmap="plasma")
    ax1.set_title("$R_{1/2,\star}^{A} > 1 && R_{1/2,\star}^{B} > 1$")
    okinds = np.logical_and(tot_hmrs <= 1, tot_prog_hmrs > 1)
    im = ax2.hexbin(delta_hmr[okinds], delta_bhms[okinds],  gridsize=30,
                    mincnt=np.min(w) - (0.1 * np.min(w)),
                    C=w[okinds], xscale="log", yscale="log",
                    reduce_C_function=np.mean, norm=weight_norm,
                    linewidths=0.2, cmap="plasma")
    ax2.set_title("$R_{1/2,\star}^{A} \leq 1 && R_{1/2,\star}^{B} > 1$")
    okinds = np.logical_and(tot_hmrs <= 1, tot_prog_hmrs <= 1)
    im = ax3.hexbin(delta_hmr[okinds], delta_bhms[okinds],  gridsize=30,
                    mincnt=np.min(w) - (0.1 * np.min(w)),
                    C=w[okinds], xscale="log", yscale="log",
                    reduce_C_function=np.mean, norm=weight_norm,
                    linewidths=0.2, cmap="plasma")
    ax3.set_title("$R_{1/2,\star}^{A} \leq 1 && R_{1/2,\star}^{B} \leq 1$")

    # Axes labels
    ax1.set_xlabel(
        "$R_\mathrm{1/2,\star}^\mathrm{B} / R_\mathrm{1/2,\star}^\mathrm{A}$")
    ax2.set_xlabel(
        "$R_\mathrm{1/2,\star}^\mathrm{B} / R_\mathrm{1/2,\star}^\mathrm{A}$")
    ax3.set_xlabel(
        "$R_\mathrm{1/2,\star}^\mathrm{B} / R_\mathrm{1/2,\star}^\mathrm{A}$")
    ax1.set_ylabel(
        "$M_\mathrm{bh}^\mathrm{B} / M_\mathrm{bh}^\mathrm{A}$")

    cbar = fig.colorbar(im, cax)
    cbar.set_label("$\sum w_{i}$")

    # Get and set universal axis limits
    xmin, xmax = np.inf, 0
    ymin, ymax = np.inf, 0
    for ax in [ax1, ax2, ax3]:
        xlims = ax.get_xlim()
        ylims = ax.get_ylim()
        if xlims[0] < xmin:
            xmin = xlims[0]
        if ylims[0] < ymin:
            ymin = ylims[0]
        if xlims[1] > xmax:
            xmax = xlims[1]
        if ylims[1] > ymax:
            ymax = ylims[1]
    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

    # Save figure
    mkdir("plots/graph/")
    fig.savefig("plots/graph/delta_hmr_blackhole.png",
                bbox_inches="tight")
    plt.close(fig)
