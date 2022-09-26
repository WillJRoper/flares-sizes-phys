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


def plot_binding_energy(data, snaps, weight_norm, comm, nranks, rank):

    # Get the dark matter mass
    hdf = h5py.File("/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/flares_00/"
                    "data/snapshot_000_z015p000/snap_000_z015p000.0.hdf5")
    mdm = hdf["Header"].attrs["MassTable"][1]
    hdf.close()

    master_base = "/cosma7/data/dp004/dc-payy1/my_files/flares_pipeline/data/flares.hdf5"

    # Open the master file
    hdf_master = h5py.File(master_base, "r")

    # Physical constants
    G = (const.G.to(u.Mpc ** 3 * u.M_sun ** -1 * u.Myr ** -2)).value

    # Loop over snapshots
    for snap in snaps:

        # Initialise lists for storing results
        tot_hmrs = []
        binding_energy = []
        feedback_energy = []
        kinetic_energy = []
        disps = []
        gal_masses = []
        w = []
        virial_param = []

        if rank == 0:
            print(snap)

        # Get redshift
        z = float(snap.split("z")[-1].replace("p", "."))

        # Define softening length
        soft = 0.001802390 / (0.6777 * (1 + z))

        # Set fake region IDs
        reg = "100"
        reg_int = -1

        if snap != snaps[-1]:
            continue

        # Extract galaxy data from the sizes dict
        hmrs = data["stellar"][snap]["HMRs"][...] / 1000
        stellar_mass = data["stellar"][snap]["mass"][...]
        regions = data["stellar"][snap]["regions"][...]
        ws = data["stellar"][snap]["weights"][...]
        grps = data["stellar"][snap]["Galaxy,GroupNumber"][...]
        subgrps = data["stellar"][snap]["Galaxy,SubGroupNumber"][...]
        if rank == 0:
            print("There are %d galaxies" % len(hmrs))
            print("There are %d compact galaxies" % len(hmrs[hmrs < 1]))

        # Loop over galaxies
        for ind in range(len(hmrs)):

            # Get the region for this galaxy
            reg_int = regions[ind]

            if int(reg) != reg_int:
                reg = str(reg_int).zfill(2)
                if rank == 0:
                    print(reg)

                if reg_int == 18:
                    continue

                # Get the master file group
                snap_grp = hdf_master[reg][snap]

                # Get the group and subgroup id arrays from the master file
                master_grps = snap_grp["Galaxy"]["GroupNumber"][...]
                master_subgrps = snap_grp["Galaxy"]["SubGroupNumber"][...]

                # Get other data from the master file
                cops = snap_grp["Galaxy"]["COP"][...].T / (1 + z)
                vel_cops = snap_grp["Galaxy"]["Velocity"][...].T
                master_s_length = snap_grp["Galaxy"]["S_Length"][...]
                master_s_pos = snap_grp["Particle"]["S_Coordinates"][...].T / \
                    (1 + z)
                master_s_vel = snap_grp["Particle"]["S_Vel"][...].T
                ini_ms = snap_grp["Particle"]["S_MassInitial"][...] * 10 ** 10
                s_mass = snap_grp["Particle"]["S_Mass"][...] * 10 ** 10
                master_g_length = snap_grp["Galaxy"]["G_Length"][...]
                master_g_pos = snap_grp["Particle"]["G_Coordinates"][...].T / \
                    (1 + z)
                master_g_vel = snap_grp["Particle"]["G_Vel"][...].T
                g_mass = snap_grp["Particle"]["G_Mass"][...] * 10 ** 10
                master_dm_length = snap_grp["Galaxy"]["DM_Length"][...]
                master_dm_pos = snap_grp["Particle"]["DM_Coordinates"][...].T / (
                    1 + z)
                master_dm_vel = snap_grp["Particle"]["DM_Vel"][...].T
                dm_mass = np.full(master_dm_pos.shape[0], mdm * 10 ** 10)
                master_bh_length = snap_grp["Galaxy"]["BH_Length"][...]
                master_bh_pos = snap_grp["Particle"]["BH_Coordinates"][...].T / (
                    1 + z)
                bh_mass = snap_grp["Particle"]["BH_Mass"][...] * 10 ** 10

            # Extract this galaxies information
            hmr = hmrs[ind]
            smass = stellar_mass[ind]
            g, sg = grps[ind], subgrps[ind]

            # Get the index in the master file
            master_ind = np.where(
                np.logical_and(master_grps == g,
                               master_subgrps == sg)
            )[0]

            if master_ind.size == 0:
                continue

            # Extract the index from the array it's contained within
            master_ind = master_ind[0]

            # Get the start index for each particle type
            s_start = np.sum(master_s_length[:master_ind])
            g_start = np.sum(master_g_length[:master_ind])
            dm_start = np.sum(master_dm_length[:master_ind])
            bh_start = np.sum(master_bh_length[:master_ind])
            s_len = master_s_length[master_ind]
            g_len = master_g_length[master_ind]
            dm_len = master_dm_length[master_ind]
            bh_len = master_bh_length[master_ind]
            cop = cops[master_ind, :]
            vel_cop = vel_cops[master_ind, :]

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
            this_s_vel = master_s_vel[s_start: s_start + s_len, :]
            this_dm_vel = master_dm_vel[dm_start: dm_start + dm_len, :]
            this_g_vel = master_g_vel[g_start: g_start + g_len, :]
            this_bh_vel = np.zeros((bh_len, 3))

            # Combine coordiantes and masses into a single array
            part_counts = [g_len,
                           dm_len,
                           0, 0,
                           s_len,
                           bh_len]
            npart = np.sum(part_counts)
            coords = np.zeros((npart, 3))
            masses = np.zeros(npart)
            vels = np.zeros((npart, 3))

            # Create lists of particle type data
            lst_coords = (this_g_pos, this_dm_pos, [], [],
                          this_s_pos, this_bh_pos)
            lst_vels = (this_g_vel, this_dm_vel, [], [],
                        this_s_vel, this_bh_vel)
            lst_masses = (this_g_mass, this_dm_mass, [], [],
                          this_s_mass, this_bh_mass)

            # Construct combined arrays
            for pos, ms, v, ipart in zip(lst_coords, lst_masses, lst_vels,
                                         range(len(part_counts))):
                if ipart > 0:
                    low = part_counts[ipart - 1]
                    high = part_counts[ipart - 1] + part_counts[ipart]
                else:
                    low, high = 0, part_counts[ipart]
                if part_counts[ipart] > 0:
                    coords[low:  high, :] = pos
                    masses[low: high] = ms
                    vels[low:  high, :] = v

            # Calculate radius and apply a 30 pkpc aperture
            rs = np.sqrt((coords[:, 0] - cop[0]) ** 2
                         + (coords[:, 1] - cop[1]) ** 2
                         + (coords[:, 2] - cop[2]) ** 2)
            okinds = rs < 0.03
            gas_rs = np.sqrt((this_g_pos[:, 0] - cop[0]) ** 2
                             + (this_g_pos[:, 1] - cop[1]) ** 2
                             + (this_g_pos[:, 2] - cop[2]) ** 2)
            g_okinds = gas_rs < 0.03
            this_g_vel = this_g_vel[g_okinds, :]

            # Get only particles within the aperture
            coords = coords[okinds, :]
            masses = masses[okinds]
            vels = vels[okinds, :] - vel_cop
            v_squ = vels[:, 0] ** 2 + vels[:, 1] ** 2 + vels[:, 2] ** 2

            # Get the particles this rank has to handle
            rank_parts = np.linspace(0, masses.size, nranks + 1, dtype=int)

            # Define gravitational binding energy
            ebind = 0

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

            # Complete the calculation
            ebind *= G / 2 * u.Msun * u.Mpc ** 2 * u.Myr ** -2

            # Convert binding energy units
            ebind = ebind.to(u.erg).value

            # Gather the results to the master
            ebind = comm.reduce(ebind, op=MPI.SUM, root=0)

            if rank == 0:

                # Calculate kinetic energy (NOTE need the hubble flow!)
                ke = np.sum(0.5 * masses * v_squ)
                ke *= u.M_sun * u.km ** 2 * u.s**-2
                ke = ke.to(u.erg).value

                # Calculate stellar radii
                star_rs = np.sqrt((this_s_pos[:, 0] - cop[0]) ** 2
                                  + (this_s_pos[:, 1] - cop[1]) ** 2
                                  + (this_s_pos[:, 2] - cop[2]) ** 2)
                s_okinds = star_rs < 0.03

                # Include these results for plotting
                tot_hmrs.append(hmr)
                feedback_energy.append(
                    np.sum(1.74 * 10 ** 49 * this_ini_ms[s_okinds]))
                binding_energy.append(ebind)
                kinetic_energy.append(ke)
                w.append(ws[ind])
                gal_masses.append(smass)
                disps.append(np.std(this_g_vel))

        if rank == 0:

            # Convert to arrays
            tot_hmrs = np.array(tot_hmrs)
            feedback_energy = np.array(feedback_energy)
            binding_energy = np.array(binding_energy)
            kinetic_energy = np.array(kinetic_energy)
            w = np.array(w)
            gal_masses = np.array(gal_masses)
            disps = np.array(disps)

            # Set up plot
            fig = plt.figure(figsize=(3.5, 3.5))
            ax = fig.add_subplot(111)
            ax.loglog()

            # Plot the scatter
            im = ax.hexbin(gal_masses, binding_energy,  gridsize=50,
                           mincnt=np.min(w) - (0.1 * np.min(w)),
                           C=w, xscale="log", yscale="log",
                           reduce_C_function=np.sum, norm=weight_norm,
                           linewidths=0.2, cmap="plasma")

            # Axes labels
            ax.set_xlabel("$M_\star / M_\odot$")
            ax.set_ylabel("$E_\mathrm{bind} / [\mathrm{erg}]$")

            cbar = fig.colorbar(im)
            cbar.set_label("$\sum w_i$")

            # Save figure
            mkdir("plots/energy/")
            fig.savefig("plots/energy/mass_energy_%s.png" % snap,
                        bbox_inches="tight")
            plt.close(fig)

            # Set up plot
            fig = plt.figure(figsize=(3.5, 3.5))
            ax = fig.add_subplot(111)
            ax.loglog()

            # Plot the scatter
            im = ax.hexbin(gal_masses, binding_energy / feedback_energy,
                           gridsize=50, mincnt=np.min(w) - (0.1 * np.min(w)),
                           C=w, xscale="log", yscale="log",
                           reduce_C_function=np.sum, norm=weight_norm,
                           linewidths=0.2, cmap="plasma")

            # Axes labels
            ax.set_xlabel("$M_\star / M_\odot$")
            ax.set_ylabel("$E_\mathrm{bind} / E_\mathrm{fb}$")

            cbar = fig.colorbar(im)
            cbar.set_label("$\sum w_i$")

            # Save figure
            mkdir("plots/energy/")
            fig.savefig("plots/energy/mass_energyratio_%s.png" % snap,
                        bbox_inches="tight")
            plt.close(fig)

            # Set up plot
            fig = plt.figure(figsize=(3.5, 3.5))
            ax = fig.add_subplot(111)
            ax.loglog()

            # Plot the scatter
            im = ax.hexbin(tot_hmrs * 1000, binding_energy / feedback_energy,
                           gridsize=50, mincnt=np.min(w) - (0.1 * np.min(w)),
                           C=w, xscale="log", yscale="log",
                           reduce_C_function=np.sum, norm=weight_norm,
                           linewidths=0.2, cmap="plasma")

            # Axes labels
            ax.set_xlabel("$R_{1/2 / [\mathrm{pkpc}]}$")
            ax.set_ylabel("$E_\mathrm{bind} / E_\mathrm{fb}$")

            cbar = fig.colorbar(im)
            cbar.set_label("$\sum w_i$")

            # Save figure
            mkdir("plots/energy/")
            fig.savefig("plots/energy/hmr_energyratio_%s.png" % snap,
                        bbox_inches="tight")
            plt.close(fig)

            # Set up plot
            fig = plt.figure(figsize=(3.5, 3.5))
            ax = fig.add_subplot(111)
            ax.loglog()

            # Plot the scatter
            im = ax.hexbin(disps, binding_energy,  gridsize=50,
                           mincnt=np.min(w) - (0.1 * np.min(w)),
                           C=w, xscale="log", yscale="log",
                           reduce_C_function=np.sum, norm=weight_norm,
                           linewidths=0.2, cmap="plasma")

            # Axes labels
            ax.set_xlabel("$\sigma_{\mathrm{gas}}$")
            ax.set_ylabel("$E_\mathrm{bind} / [\mathrm{erg}]$")

            cbar = fig.colorbar(im)
            cbar.set_label("$\sum w_i$")

            # Save figure
            mkdir("plots/energy/")
            fig.savefig("plots/energy/disp_energy_%s.png" % snap,
                        bbox_inches="tight")
            plt.close(fig)

            # Set up plot
            fig = plt.figure(figsize=(3.5, 3.5))
            ax = fig.add_subplot(111)
            ax.loglog()

            okinds = np.logical_and(kinetic_energy > 0, binding_energy > 0)

            # Plot the scatter
            im = ax.hexbin(gal_masses[okinds],
                           kinetic_energy[okinds] / binding_energy[okinds],
                           gridsize=50, mincnt=np.min(w) - (0.1 * np.min(w)),
                           C=w[okinds], xscale="log", yscale="log",
                           reduce_C_function=np.sum, norm=weight_norm,
                           linewidths=0.2, cmap="plasma")

            # Axes labels
            ax.set_xlabel("$M_\star / M_\odot$")
            ax.set_ylabel("$E_{\mathrm{KE}} / E_\mathrm{bind}$")

            cbar = fig.colorbar(im)
            cbar.set_label("$\sum w_i$")

            # Save figure
            mkdir("plots/energy/")
            fig.savefig("plots/energy/mass_kinenergyratio_%s.png" % snap,
                        bbox_inches="tight")
            plt.close(fig)

            # Set up plot
            fig = plt.figure(figsize=(3.5, 3.5))
            ax = fig.add_subplot(111)
            ax.loglog()

            # Plot the scatter
            im = ax.hexbin(tot_hmrs * 1000, kinetic_energy / binding_energy,
                           gridsize=50, mincnt=np.min(w) - (0.1 * np.min(w)),
                           C=w, xscale="log", yscale="log",
                           reduce_C_function=np.sum, norm=weight_norm,
                           linewidths=0.2, cmap="plasma")

            # Axes labels
            ax.set_xlabel("$R_{1/2 / [\mathrm{pkpc}]}$")
            ax.set_ylabel("$E_{\mathrm{KE}} / E_\mathrm{bind}$")

            cbar = fig.colorbar(im)
            cbar.set_label("$\sum w_i$")

            # Save figure
            mkdir("plots/energy/")
            fig.savefig("plots/energy/hmr_kinenergyratio_%s.png" % snap,
                        bbox_inches="tight")
            plt.close(fig)

    hdf_master.close()


def plot_virial_param(data, snaps, weight_norm):

    # Get the dark matter mass
    hdf = h5py.File("/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/flares_00/"
                    "data/snapshot_000_z015p000/snap_000_z015p000.0.hdf5")
    mdm = hdf["Header"].attrs["MassTable"][1]
    hdf.close()

    master_base = "/cosma7/data/dp004/dc-payy1/my_files/flares_pipeline/data/flares.hdf5"

    # Open the master file
    hdf_master = h5py.File(master_base, "r")

    # Physical constants
    G = (const.G.to(u.Mpc ** 3 * u.M_sun ** -1 * u.Myr ** -2)).value

    # Loop over snapshots
    for snap in snaps:

        # Initialise lists for storing results
        tot_hmrs = []
        disps = []
        gal_masses = []
        w = []
        virial_param = []

        print(snap)

        # Get redshift
        z = float(snap.split("z")[-1].replace("p", "."))

        # Define softening length
        soft = 0.001802390 / (0.6777 * (1 + z))

        # Set fake region IDs
        reg = "100"
        reg_int = -1

        # Extract galaxy data from the sizes dict
        hmrs = data["stellar"][snap]["HMRs"][...] / 1000
        stellar_mass = data["stellar"][snap]["mass"][...]
        regions = data["stellar"][snap]["regions"][...]
        ws = data["stellar"][snap]["weights"][...]
        grps = data["stellar"][snap]["Galaxy,GroupNumber"][...]
        subgrps = data["stellar"][snap]["Galaxy,SubGroupNumber"][...]
        print("There are %d galaxies" % len(hmrs))
        print("There are %d compact galaxies" % len(hmrs[hmrs < 1]))

        # Loop over galaxies
        for ind in range(len(hmrs)):

            # Get the region for this galaxy
            reg_int = regions[ind]

            if int(reg) != reg_int:
                reg = str(reg_int).zfill(2)
                print(reg)

                if reg_int == 18:
                    continue

                # Get the master file group
                snap_grp = hdf_master[reg][snap]

                # Get the group and subgroup id arrays from the master file
                master_grps = snap_grp["Galaxy"]["GroupNumber"][...]
                master_subgrps = snap_grp["Galaxy"]["SubGroupNumber"][...]

                # Get other data from the master file
                cops = snap_grp["Galaxy"]["COP"][...].T / (1 + z)
                master_s_length = snap_grp["Galaxy"]["S_Length"][...]
                master_s_pos = snap_grp["Particle"]["S_Coordinates"][...].T / \
                    (1 + z)
                master_s_vel = snap_grp["Particle"]["S_Vel"][...].T
                ini_ms = snap_grp["Particle"]["S_MassInitial"][...] * 10 ** 10
                s_mass = snap_grp["Particle"]["S_Mass"][...] * 10 ** 10
                master_g_length = snap_grp["Galaxy"]["G_Length"][...]
                master_g_pos = snap_grp["Particle"]["G_Coordinates"][...].T / \
                    (1 + z)
                master_g_vel = snap_grp["Particle"]["G_Vel"][...].T
                g_mass = snap_grp["Particle"]["G_Mass"][...] * 10 ** 10
                master_dm_length = snap_grp["Galaxy"]["DM_Length"][...]
                master_dm_pos = snap_grp["Particle"]["DM_Coordinates"][...].T / (
                    1 + z)
                master_dm_vel = snap_grp["Particle"]["DM_Vel"][...].T
                dm_mass = np.full(master_dm_pos.shape[0], mdm * 10 ** 10)
                master_bh_length = snap_grp["Galaxy"]["BH_Length"][...]
                master_bh_pos = snap_grp["Particle"]["BH_Coordinates"][...].T / (
                    1 + z)
                bh_mass = snap_grp["Particle"]["BH_Mass"][...] * 10 ** 10

            # Extract this galaxies information
            hmr = hmrs[ind]
            smass = stellar_mass[ind]
            g, sg = grps[ind], subgrps[ind]

            # Get the index in the master file
            master_ind = np.where(
                np.logical_and(master_grps == g,
                               master_subgrps == sg)
            )[0]

            if master_ind.size == 0:
                continue

            # Extract the index from the array it's contained within
            master_ind = master_ind[0]

            # Get the start index for each particle type
            s_start = np.sum(master_s_length[:master_ind])
            g_start = np.sum(master_g_length[:master_ind])
            dm_start = np.sum(master_dm_length[:master_ind])
            bh_start = np.sum(master_bh_length[:master_ind])
            s_len = master_s_length[master_ind]
            g_len = master_g_length[master_ind]
            dm_len = master_dm_length[master_ind]
            bh_len = master_bh_length[master_ind]
            cop = cops[master_ind, :]

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
            this_s_vel = master_s_vel[s_start: s_start + s_len, :]
            this_dm_vel = master_dm_vel[dm_start: dm_start + dm_len, :]
            this_g_vel = master_g_vel[g_start: g_start + g_len, :]
            this_bh_vel = np.full((bh_len, 3), np.nan)

            # Combine coordiantes and masses into a single array
            part_counts = [g_len,
                           dm_len,
                           0, 0,
                           s_len,
                           bh_len]
            npart = np.sum(part_counts)
            coords = np.zeros((npart, 3))
            masses = np.zeros(npart)
            vels = np.zeros((npart, 3))

            # Create lists of particle type data
            lst_coords = (this_g_pos, this_dm_pos, [], [],
                          this_s_pos, this_bh_pos)
            lst_vels = (this_g_vel, this_dm_vel, [], [],
                        this_s_vel, this_bh_vel)
            lst_masses = (this_g_mass, this_dm_mass, [], [],
                          this_s_mass, this_bh_mass)

            # Construct combined arrays
            for pos, ms, v, ipart in zip(lst_coords, lst_masses, lst_vels,
                                         range(len(part_counts))):
                if ipart > 0:
                    low = part_counts[ipart - 1]
                    high = part_counts[ipart - 1] + part_counts[ipart]
                else:
                    low, high = 0, part_counts[ipart]
                if part_counts[ipart] > 0:
                    coords[low:  high, :] = pos
                    masses[low: high] = ms
                    vels[low:  high, :] = v

            # Calculate radius and apply a 30 pkpc aperture
            rs = np.sqrt((coords[:, 0] - cop[0]) ** 2
                         + (coords[:, 1] - cop[1]) ** 2
                         + (coords[:, 2] - cop[2]) ** 2)
            gas_rs = np.sqrt((this_g_pos[:, 0] - cop[0]) ** 2
                             + (this_g_pos[:, 1] - cop[1]) ** 2
                             + (this_g_pos[:, 2] - cop[2]) ** 2)
            star_rs = np.sqrt((this_s_pos[:, 0] - cop[0]) ** 2
                              + (this_s_pos[:, 1] - cop[1]) ** 2
                              + (this_s_pos[:, 2] - cop[2]) ** 2)

            # Include these results for plotting
            tot_hmrs.append(hmr)
            w.append(ws[ind])
            gal_masses.append(smass)
            disps.append(np.std(this_g_vel))
            okinds = rs < 0.001
            virial_param.append(
                5 * np.nanstd(
                    (vels[okinds] * u.km / u.s).to(u.Mpc / u.Myr).value)**2
                * 2 * hmr / (G * np.sum(masses[okinds]))
            )

        # Convert to arrays
        tot_hmrs = np.array(tot_hmrs)
        w = np.array(w)
        gal_masses = np.array(gal_masses)
        disps = np.array(disps)
        virial_params = np.array(virial_param)

        # Set up plot
        fig = plt.figure(figsize=(3.5, 3.5))
        ax = fig.add_subplot(111)
        ax.loglog()

        # Plot the scatter
        im = ax.hexbin(gal_masses, virial_params,  gridsize=50,
                       mincnt=np.min(w) - (0.1 * np.min(w)),
                       C=w, xscale="log", yscale="log",
                       reduce_C_function=np.sum, norm=weight_norm,
                       linewidths=0.2, cmap="plasma")

        # Axes labels
        ax.set_xlabel("$M_\star / M_\odot$")
        ax.set_ylabel(r"$\alpha$")

        cbar = fig.colorbar(im)
        cbar.set_label("$\sum w_i$")

        # Save figure
        mkdir("plots/energy/")
        fig.savefig("plots/energy/mass_virparam_%s.png" % snap,
                    bbox_inches="tight")
        plt.close(fig)

    hdf_master.close()


def plot_virial_param_profile(data, snaps, weight_norm):

    # Get the dark matter mass
    hdf = h5py.File("/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/flares_00/"
                    "data/snapshot_000_z015p000/snap_000_z015p000.0.hdf5")
    mdm = hdf["Header"].attrs["MassTable"][1]
    hdf.close()

    master_base = "/cosma7/data/dp004/dc-payy1/my_files/flares_pipeline/data/flares.hdf5"

    # Open the master file
    hdf_master = h5py.File(master_base, "r")

    # Physical constants
    G = (const.G.to(u.Mpc ** 3 * u.M_sun ** -1 * u.Myr ** -2)).value

    # Define radial bins
    r_bins = np.logspace(-5, np.log10(30), 100)
    rbin_cents = (r_bins[:-1] + r_bins[1:]) / 2

    # Define mass bins
    m_bins = np.logspace(8, 11.5, 50)

    # Loop over snapshots
    for snap in snaps:

        # Initialise lists for storing results
        tot_hmrs = []
        gal_masses = []
        w = []
        virial_params = []
        prof_rs = []

        print(snap)

        # Get redshift
        z = float(snap.split("z")[-1].replace("p", "."))

        # Define softening length
        soft = 0.001802390 / (0.6777 * (1 + z))

        # Set fake region IDs
        reg = "100"
        reg_int = -1

        # Extract galaxy data from the sizes dict
        hmrs = data["stellar"][snap]["HMRs"][...] / 1000
        stellar_mass = data["stellar"][snap]["mass"][...]
        regions = data["stellar"][snap]["regions"][...]
        ws = data["stellar"][snap]["weights"][...]
        grps = data["stellar"][snap]["Galaxy,GroupNumber"][...]
        subgrps = data["stellar"][snap]["Galaxy,SubGroupNumber"][...]

        # Loop over galaxies
        for ind in range(len(hmrs)):

            # Get the region for this galaxy
            reg_int = regions[ind]

            if int(reg) != reg_int:
                reg = str(reg_int).zfill(2)
                print(reg)

                if reg_int == 18:
                    continue

                # Get the master file group
                snap_grp = hdf_master[reg][snap]

                # Get the group and subgroup id arrays from the master file
                master_grps = snap_grp["Galaxy"]["GroupNumber"][...]
                master_subgrps = snap_grp["Galaxy"]["SubGroupNumber"][...]

                # Get other data from the master file
                cops = snap_grp["Galaxy"]["COP"][...].T / (1 + z)
                master_s_length = snap_grp["Galaxy"]["S_Length"][...]
                master_s_pos = snap_grp["Particle"]["S_Coordinates"][...].T / \
                    (1 + z)
                master_s_vel = snap_grp["Particle"]["S_Vel"][...].T
                ini_ms = snap_grp["Particle"]["S_MassInitial"][...] * 10 ** 10
                s_mass = snap_grp["Particle"]["S_Mass"][...] * 10 ** 10
                master_g_length = snap_grp["Galaxy"]["G_Length"][...]
                master_g_pos = snap_grp["Particle"]["G_Coordinates"][...].T / \
                    (1 + z)
                master_g_vel = snap_grp["Particle"]["G_Vel"][...].T
                g_mass = snap_grp["Particle"]["G_Mass"][...] * 10 ** 10
                master_dm_length = snap_grp["Galaxy"]["DM_Length"][...]
                master_dm_pos = snap_grp["Particle"]["DM_Coordinates"][...].T / (
                    1 + z)
                master_dm_vel = snap_grp["Particle"]["DM_Vel"][...].T
                dm_mass = np.full(master_dm_pos.shape[0], mdm * 10 ** 10)
                master_bh_length = snap_grp["Galaxy"]["BH_Length"][...]
                master_bh_pos = snap_grp["Particle"]["BH_Coordinates"][...].T / (
                    1 + z)
                bh_mass = snap_grp["Particle"]["BH_Mass"][...] * 10 ** 10

            # Extract this galaxies information
            hmr = hmrs[ind]
            smass = stellar_mass[ind]
            g, sg = grps[ind], subgrps[ind]

            # Get the index in the master file
            master_ind = np.where(
                np.logical_and(master_grps == g,
                               master_subgrps == sg)
            )[0]

            if master_ind.size == 0:
                continue

            # Extract the index from the array it's contained within
            master_ind = master_ind[0]

            # Get the start index for each particle type
            s_start = np.sum(master_s_length[:master_ind])
            g_start = np.sum(master_g_length[:master_ind])
            dm_start = np.sum(master_dm_length[:master_ind])
            bh_start = np.sum(master_bh_length[:master_ind])
            s_len = master_s_length[master_ind]
            g_len = master_g_length[master_ind]
            dm_len = master_dm_length[master_ind]
            bh_len = master_bh_length[master_ind]
            cop = cops[master_ind, :]

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
            this_s_vel = master_s_vel[s_start: s_start + s_len, :]
            this_dm_vel = master_dm_vel[dm_start: dm_start + dm_len, :]
            this_g_vel = master_g_vel[g_start: g_start + g_len, :]
            this_bh_vel = np.full((bh_len, 3), np.nan)

            # Combine coordiantes and masses into a single array
            part_counts = [g_len,
                           dm_len,
                           0, 0,
                           s_len,
                           bh_len]
            npart = np.sum(part_counts)
            coords = np.zeros((npart, 3))
            masses = np.zeros(npart)
            vels = np.zeros((npart, 3))

            # Create lists of particle type data
            lst_coords = (this_g_pos, this_dm_pos, [], [],
                          this_s_pos, this_bh_pos)
            lst_vels = (this_g_vel, this_dm_vel, [], [],
                        this_s_vel, this_bh_vel)
            lst_masses = (this_g_mass, this_dm_mass, [], [],
                          this_s_mass, this_bh_mass)

            # Construct combined arrays
            for pos, ms, v, ipart in zip(lst_coords, lst_masses, lst_vels,
                                         range(len(part_counts))):
                if ipart > 0:
                    low = part_counts[ipart - 1]
                    high = part_counts[ipart - 1] + part_counts[ipart]
                else:
                    low, high = 0, part_counts[ipart]
                if part_counts[ipart] > 0:
                    coords[low:  high, :] = pos
                    masses[low: high] = ms
                    vels[low:  high, :] = v

            # Calculate radius and apply a 30 pkpc aperture
            rs = np.sqrt((coords[:, 0] - cop[0]) ** 2
                         + (coords[:, 1] - cop[1]) ** 2
                         + (coords[:, 2] - cop[2]) ** 2)

            # Sort particles
            sinds = np.argsort(rs)
            rs = rs[sinds]
            vels = vels[sinds]
            masses = masses[sinds]

            # Loop over radial bins
            for r_ind, r in enumerate(rbin_cents):

                r_okinds = rs < r

                # Include these results for plotting
                w.append(ws[ind])
                gal_masses.append(smass)
                virial_params.append(
                    5 * np.nanstd(
                        (vels[r_okinds] * u.km / u.s).to(u.Mpc / u.Myr).value)**2
                    * r / (G * np.sum(masses[r_okinds]))
                )
                prof_rs.append(r)

        # Convert to arrays
        w = np.array(w)
        gal_masses = np.array(gal_masses)
        virial_params = np.array(virial_params)
        prof_rs = np.array(prof_rs) * 1000

        # Define the mass normalisation for the colormap
        norm = cm.TwoSlopeNorm(vmin=8, vcenter=9, vmax=11.5)
        cmap = plt.get_cmap("cmr.guppy")

        # Set up plot
        fig = plt.figure(figsize=(3.5, 3.5))
        ax = fig.add_subplot(111)
        ax.loglog()

        # Loop over mass bins
        for i in range(m_bins.size - 1):

            # Get bin edges
            mlow, mhigh = m_bins[i], m_bins[i + 1]

            # Define bin midpoint for color
            m_cent = (mlow + mhigh) / 2

            # Define mask for this mass bin_cents
            okinds = np.logical_and(gal_masses >= mlow,
                                    gal_masses < mhigh)

            if len(w[okinds]) == 0:
                continue

            # Plot this profile
            plot_meidan_stat(prof_rs[okinds],
                             virial_params[okinds], w[okinds],
                             ax, lab="", color=cmap(norm(np.log10(m_cent))),
                             bins=r_bins)

        # Axes labels
        ax.set_xlabel("$R / [\mathrm{pkpc}]$")
        ax.set_ylabel(r"$\alpha(<R)$")

        cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap))
        cbar.set_label("$M_\star / M_\odot$")

        # Save figure
        mkdir("plots/energy/")
        fig.savefig("plots/energy/mass_virparam_profile_%s.png" % snap,
                    bbox_inches="tight")
        plt.close(fig)

    hdf_master.close()
