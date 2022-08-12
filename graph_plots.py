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


def plot_size_change(stellar_data, snaps):

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

    # Loop over snapshots
    for snap, prog_snap in zip(current_snaps, prog_snaps):

        # Open region 0 initially
        reg = "00"
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
        mega_prog_grps = hdf_prog["subgroup_number"][...]
        mega_prog_subgrps = hdf_prog["subgroup_number"][...]

        # Get the progenitor information
        prog_mass_conts = hdf_graph["ProgMassContribution"][...]
        prog_ids = hdf_graph["ProgHaloIDs"][...]
        start_index = hdf_graph["prog_start_index"][...]
        nprogs = hdf_graph["n_progs"][...]

        hdf_halo.close()
        hdf_prog.close()
        hdf_graph.close()

        # Extract galaxy data from the sizes dict
        hmrs = stellar_data[snap]["HMRs"]
        prog_hmrs = stellar_data[prog_snap]["HMRs"]
        grps = stellar_data[snap]["Galaxy,GroupNumber"]
        subgrps = stellar_data[snap]["Galaxy,SubGroupNumber"]
        prog_grps = stellar_data[prog_snap]["Galaxy,GroupNumber"]
        prog_subgrps = stellar_data[prog_snap]["Galaxy,SubGroupNumber"]
        regions = stellar_data[snap]["regions"]
        prog_regions = stellar_data[prog_snap]["regions"]

        # Get only this regions flares data
        reg_okinds = prog_regions == reg_int
        reg_prog_hmrs = prog_hmrs[reg_okinds]
        reg_prog_grps = prog_grps[reg_okinds]
        reg_prog_subgrps = prog_subgrps[reg_okinds]

        # Loop over galaxies
        for ind in range(len(hmrs)):

            # Skip this galaxy if it is not compact
            if hmrs[ind] > 1:
                continue

            if reg_int == 18:
                continue

            # Get the region for this galaxy
            reg_int = regions[ind]
            if int(reg) != reg_int:
                reg = str(reg_int).zfill(2)

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
                mega_prog_grps = hdf_prog["subgroup_number"][...]
                mega_prog_subgrps = hdf_prog["subgroup_number"][...]

                # Get the progenitor information
                prog_mass_conts = hdf_graph["ProgMassContribution"][...]
                prog_ids = hdf_graph["ProgHaloIDs"][...]
                start_index = hdf_graph["prog_start_index"][...]
                nprogs = hdf_graph["n_progs"][...]

                hdf_halo.close()
                hdf_prog.close()
                hdf_graph.close()

                # Get only this regions flares data
                reg_okinds = prog_regions == reg_int
                reg_prog_hmrs = prog_hmrs[reg_okinds]
                reg_prog_grps = prog_grps[reg_okinds]
                reg_prog_subgrps = prog_subgrps[reg_okinds]

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

            # Get this progenitors group and subgroup ID
            prog_g = mega_prog_grps[main_prog]
            prog_sg = mega_prog_subgrps[main_prog]

            # Get this progenitor's size
            flares_ind = np.where(
                np.logical_and(reg_prog_grps == prog_g,
                               reg_prog_subgrps == prog_sg)
            )[0]
            prog_hmr = reg_prog_hmrs[flares_ind]
            print(prog_g, prog_sg, flares_ind, prog_hmr)

            if prog_hmr.size == 0:
                continue

            # Get the contribution information
            prog_cont = prog_mass_conts[start: start + stride] * 10 ** 10
            mass = masses[mega_ind] * 10 ** 10

            # Calculate the mass contribution as a fraction of current mass
            tot_prog_cont = np.sum(prog_cont)
            frac_prog_cont = tot_prog_cont / mass

            # Include these results for plotting
            tot_cont.extend(frac_prog_cont)
            tot_hmrs.extend(hmr)
            tot_prog_hmrs.extend(prog_hmr)

    # Convert to arrays
    print(tot_hmrs)
    print(tot_cont)
    print(tot_prog_hmrs)
    tot_hmrs = np.array(tot_hmrs)
    tot_prog_hmrs = np.array(tot_prog_hmrs)
    tot_cont = np.array(tot_cont)

    # Compute delta
    delta_hmr = tot_hmrs - tot_prog_hmrs

    # Set up plot
    fig = plt.figure(figsize=(3.5, 3.5))
    ax = fig.add_subplot(111)

    # Plot the scatter
    ax.scatter(tot_cont, delta_hmr, marker=".")

    # Axes labels
    ax.set_xlabel("$M_\mathrm{cont} / M_\mathrm{tot}$")
    ax.set_ylabel("$\Delta R_{1/2}$")

    # Save figure
    mkdir("plots/graph/")
    fig.savefig("plots/graph/delta_hmr_contribution.png",
                bbox_inches="tight", dpi=100)
