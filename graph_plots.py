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
    w = []

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
        ws = stellar_data[snap]["weights"]

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
            w.append(ws[ind])

    # Convert to arrays
    gas_tot_hmrs = np.array(tot_hmrs["gas"])
    gas_tot_prog_hmrs = np.array(tot_prog_hmrs["gas"])
    star_tot_hmrs = np.array(tot_hmrs["star"])
    star_tot_prog_hmrs = np.array(tot_prog_hmrs["star"])
    tot_cont = np.array(tot_cont)
    w = np.array(w)

    # Get deltas
    gas_delta_hmr = gas_tot_hmrs - gas_tot_prog_hmrs
    star_delta_hmr = star_tot_hmrs - star_tot_prog_hmrs

    # Set up plot
    fig = plt.figure(figsize=(3.5, 3.5))
    ax = fig.add_subplot(111)

    # Plot the scatter
    im = ax.hexbin(gas_delta_hmr, star_delta_hmr, gridsize=30,
                   mincnt=np.min(w) - (0.1 * np.min(w)),
                   C=w,
                   reduce_C_function=np.sum,
                   linewidths=0.2, norm=weight_norm, cmap="plasma")

    # Axes labels
    ax.set_xlabel("$\Delta R_\mathrm{gas} / [\mathrm{pkpc}]$")
    ax.set_ylabel("$\Delta R_\star / [\mathrm{pkpc}]$")

    cbar = fig.colorbar(im)
    cbar.set_label("$\sum w_{i}$")

    # Save figure
    mkdir("plots/graph/")
    fig.savefig("plots/graph/delta_hmr_comp.png",
                bbox_inches="tight")
    plt.close(fig)


def plot_size_change_binding(stellar_data, snaps, weight_norm):

    # Get the dark matter mass
    hdf = h5py.File("/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/G-EAGLE_00/"
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
    G = (const.G.to(u.Mpc ** 3 * u.M_sun ** -1 * u.yr ** -2))

    # Loop over snapshots
    for snap, prog_snap in zip(current_snaps, prog_snaps):

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

                # Get the master file group
                snap_grp = hdf_master[reg][snap]
                prog_grp = hdf_master[reg][prog_snap]

                # Get the group and subgroup id arrays from the master file
                master_grps = snap_grp["Galaxy"]["GroupNumber"][...]
                master_subgrps = snap_grp["Galaxy"]["SubGroupNumber"][...]
                prog_master_grps = prog_grp["Galaxy"]["GroupNumber"][...]
                prog_master_subgrps = prog_grp["Galaxy"]["SubGroupNumber"][...]

                # Get other data from the master file
                cops = snap_grp["Galaxy"]["COP"][...]
                master_s_length = snap_grp["Galaxy"]["S_Length"][...]
                master_s_pos = snap_grp["Particle"]["S_Coordinates"][...]
                ini_ms = snap_grp["Particle"]["S_MassInitial"][...]
                s_mass = snap_grp["Particle"]["S_Mass"][...]
                master_g_length = snap_grp["Galaxy"]["G_Length"][...]
                master_g_pos = snap_grp["Particle"]["G_Coordinates"][...]
                g_mass = snap_grp["Particle"]["G_Mass"][...]
                master_dm_length = snap_grp["Galaxy"]["DM_Length"][...]
                master_dm_pos = snap_grp["Particle"]["DM_Coordinates"][...]
                dm_mass = np.full(master_dm_pos.shape[0], mdm)
                master_bh_length = snap_grp["Galaxy"]["BH_Length"][...]
                master_bh_pos = snap_grp["Particle"]["BH_Coordinates"][...]
                bh_mass = snap_grp["Particle"]["BH_Mass"][...]
                prog_cops = prog_grp["Galaxy"]["COP"][...]
                prog_master_s_length = prog_grp["Galaxy"]["S_Length"][...]
                prog_master_s_pos = prog_grp["Particle"]["S_Coordinates"][...]
                prog_ini_ms = prog_grp["Particle"]["S_MassInitial"][...]
                prog_s_mass = prog_grp["Particle"]["S_Mass"][...]
                prog_master_g_length = prog_grp["Galaxy"]["G_Length"][...]
                prog_master_g_pos = prog_grp["Particle"]["G_Coordinates"][...]
                prog_g_mass = prog_grp["Particle"]["G_Mass"][...]
                prog_master_dm_length = prog_grp["Galaxy"]["DM_Length"][...]
                prog_master_dm_pos = prog_grp["Particle"]["DM_Coordinates"][...]
                prog_dm_mass = np.full(prog_master_dm_pos.shape[0], mdm)
                prog_master_bh_length = prog_grp["Galaxy"]["BH_Length"][...]
                prog_master_bh_pos = prog_grp["Particle"]["BH_Coordinates"][...]
                prog_bh_mass = prog_grp["Particle"]["BH_Mass"][...]

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

            # Get the start index for each particle type
            s_start = np.sum(master_s_length[:master_ind])
            g_start = np.sum(master_g_length[:master_ind])
            dm_start = np.sum(master_dm_length[:master_ind])
            bh_start = np.sum(master_bh_length[:master_ind])
            s_len = master_s_length[master_ind]
            g_len = master_g_length[master_ind]
            dm_len = master_dm_length[master_ind]
            bh_len = master_bh_length[master_ind]

            prog_s_start = np.sum(prog_master_s_length[:prog_master_ind])
            prog_g_start = np.sum(prog_master_g_length[:prog_master_ind])
            prog_dm_start = np.sum(prog_master_dm_length[:prog_master_ind])
            prog_bh_start = np.sum(prog_master_bh_length[:prog_master_ind])
            prog_s_len = prog_master_s_length[prog_master_ind]
            prog_g_len = prog_master_g_length[prog_master_ind]
            prog_dm_len = prog_master_dm_length[prog_master_ind]
            prog_bh_len = prog_master_bh_length[prog_master_ind]

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
            coords = np.concatenate((this_bh_pos, this_s_pos,
                                     this_g_pos, this_dm_pos))
            prog_coords = np.concatenate((prog_this_bh_pos, prog_this_s_pos,
                                          prog_this_g_pos, prog_this_dm_pos))
            masses = np.concatenate((this_bh_mass, this_s_mass,
                                     this_g_mass, this_dm_mass))
            prog_masses = np.concatenate((prog_this_bh_mass, prog_this_s_mass,
                                          prog_this_g_mass, prog_this_dm_mass))

            # Calcualte the binding energy
            ebind = grav(coords, soft, masses, z, G)
            prog_ebind = grav(prog_coords, prog_soft, prog_masses, prog_z, G)

            # Include these results for plotting
            tot_hmrs.append(hmr)
            tot_prog_hmrs.extend(prog_hmr)
            feedback_energy.append(np.sum(1.74 * 10 ** 49 * this_ini_ms))
            prog_feedback_energy.append(np.sum(1.74 * 10 ** 49 *
                                               prog_this_ini_ms))
            binding_energy.append(ebind)
            prog_binding_energy.append(prog_ebind)
            w.append(ws[ind])

    hdf_master.close()

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

    # Set up plot
    fig = plt.figure(figsize=(3.5, 3.5))
    ax = fig.add_subplot(111)

    # Plot the scatter
    im = ax.scatter(delta_eb, delta_fb, c=delta_hmr, cmap="plasma", marker=".")

    # Axes labels
    ax.set_xlabel("$E_\mathrm{fb}^\mathrm{B} / E_\mathrm{fb}^\mathrm{A}$")
    ax.set_ylabel("$E_\mathrm{bind}^\mathrm{B} / E_\mathrm{bind}^\mathrm{A}$")

    cbar = fig.colorbar(im)
    cbar.set_label("$\Delta R_{1/2} / [\mathrm{pkpc}]$")

    # Save figure
    mkdir("plots/graph/")
    fig.savefig("plots/graph/delta_hmr_bind.png",
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
    root_hmrs = stellar_data[root_snap]["HMRs"]
    root_mass = stellar_data[root_snap]["mass"]
    root_grps = stellar_data[root_snap]["Galaxy,GroupNumber"]
    root_subgrps = stellar_data[root_snap]["Galaxy,SubGroupNumber"]
    root_regions = stellar_data[root_snap]["regions"]

    # Loop over galaxies and populate the root level of the graph with
    # only the compact galaxies
    for ind in range(len(root_hmrs)):

        # Skip if the galaxy isn't compact
        if root_hmrs[ind] > 1 or root_mass[ind] < 10 ** 10:
            continue

        # Get ID
        g, sg = root_grps[ind], root_subgrps[ind]

        # Make an entry in the dict for it
        graph[(g, sg, ind)] = {"HMRs": [], "Masses": []}

    # Loop over these root galaxies and populate the rest of the graph
    i += 1
    for key in graph:

        print("Walking %d (%d, %d, %d) of %d" % (i, key[0], key[1],
                                                 key[2], len(graph)), end="\r")
        print("", end="\r")

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

        # Loop until we don't have a progenitor to step to
        while snap_ind >= 0:

            # Extract galaxy data from the sizes dict
            hmrs = stellar_data[snap]["HMRs"]
            mass = stellar_data[snap]["mass"]
            prog_grps = stellar_data[prog_snap]["Galaxy,GroupNumber"]
            prog_subgrps = stellar_data[prog_snap]["Galaxy,SubGroupNumber"]

            # Put this galaxy in the graph
            graph[(g, sg, ind)]["HMRs"].append(hmrs[this_ind])
            graph[(g, sg, ind)]["Masses"].append(mass[this_ind])

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

            hdf_halo.close()
            hdf_prog.close()
            hdf_graph.close()

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
                this_ind = np.where(np.logical_and(prog_grps == this_g,
                                                   prog_subgrps == this_sg))[0]

                if this_ind.size == 0:
                    break

                # Set snapshots
                snap_ind -= 1
                snap = snaps[snap_ind]
                prog_snap = snaps[snap_ind - 1]

            else:
                break

    # Loop over graphs
    i = 0
    for key in graph:

        print("Plotting %d (%d, %d, %d) of %d" % (i, key[0], key[1],
                                                  key[2], len(graph)), end="\r")
        print("", end="\r")

        # Set up plot
        fig = plt.figure(figsize=(3.5, 3.5))
        ax = fig.add_subplot(111)

        # Plot the scatter
        im = ax.plot(graph[key]["Masses"], graph[key]["HMRs"], marker=".")

        # Axes labels
        ax.set_xlabel("$M_\star / M_\odot$")
        ax.set_ylabel("$R_{1/2} / [\mathrm{pkpc}]$")

        # Save figure
        mkdir("plots/graph_plot/")
        fig.savefig("plots/graph_plot/size_mass_evo_%s_%s.png" % (key[0],
                                                                  key[1]),
                    bbox_inches="tight")
        plt.close(fig)
        i += 1
