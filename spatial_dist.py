import numpy as np
import matplotlib as mpl
import matplotlib.colors as cm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from flare import plt as flareplt
from utils import mkdir, plot_meidan_stat, calc_ages
from astropy.cosmology import Planck18 as cosmo
from astropy.cosmology import z_at_value
import astropy.units as u
import eagle_IO.eagle_IO as eagle_io


def sfr_radial_profile(stellar_data, snaps, eagle_path):

    # Define radial bins
    radial_bins = np.arange(0, 30.25, 0.25)
    bin_cents = (radial_bins[:-1] + radial_bins[1:]) / 2

    # Define redshift colormap and normalisation
    cmap = mpl.cm.get_cmap('plasma')
    norm = cm.Normalize(vmin=0, vmax=10)

    # Set up plot
    fig = plt.figure(figsize=(2.75, 2.75))
    gs = gridspec.GridSpec(nrows=1, ncols=1 + 1,
                           width_ratios=[20, ] + [1, ])
    gs.update(wspace=0.0, hspace=0.0)
    ax = fig.add_subplot(gs[0, 0])
    ax.semilogy()
    cax = fig.add_subplot(gs[0, 1])

    # Loop over snapshots
    for snap in snaps:

        print("Plotting sfr radial profile for snap", snap)

        # Get redshift
        z = float(snap.split("z")[-1].replace("p", "."))

        # Calculate redshift 100 Myrs before this z
        z_100 = z_at_value(cosmo.age, cosmo.age(z) - 201 * u.Myr)

        # Initialise lists to hold the sfr profiles
        sfr_profile = []
        all_radii = []

        # Are we dealing with EAGLE or FLARES?
        if z < 5:

            aborn = eagle_io.read_array('PARTDATA', eagle_path, snap,
                                        'PartType4/StellarFormationTime',
                                        noH=True,
                                        physicalUnits=True,
                                        numThreads=8)
            ini_ms = eagle_io.read_array('PARTDATA', eagle_path, snap,
                                         'PartType4/InitialMass', noH=True,
                                         physicalUnits=True,
                                         numThreads=8) * 10 ** 10
            gal_ms = eagle_io.read_array('SUBFIND', eagle_path, snap,
                                         'Subhalo/ApertureMeasurements/Mass/030kpc', noH=True,
                                         physicalUnits=True,
                                         numThreads=8)[:, 4] * 10 ** 10
            pos = eagle_io.read_array('PARTDATA', eagle_path, snap,
                                      'PartType4/Coordinates', noH=True,
                                      physicalUnits=True,
                                      numThreads=8) * 10**3
            cops = eagle_io.read_array('SUBFIND', eagle_path, snap,
                                       'Subhalo/CentreOfPotential', noH=True,
                                       physicalUnits=True,
                                       numThreads=8) * 10**3
            part_subgrp = eagle_io.read_array('PARTDATA', eagle_path, snap,
                                              'PartType4/SubGroupNumber',
                                              noH=True,
                                              physicalUnits=True,
                                              numThreads=8)
            part_grp = eagle_io.read_array('PARTDATA', eagle_path, snap,
                                           'PartType4/GroupNumber', noH=True,
                                           physicalUnits=True,
                                           numThreads=8)
            hmrs = eagle_io.read_array('SUBFIND', eagle_path, snap,
                                       'Subhalo/HalfMassRad', noH=True,
                                       physicalUnits=True,
                                       numThreads=8)[:, 4] * 10**3
            subgrps = eagle_io.read_array('SUBFIND', eagle_path, snap,
                                          'Subhalo/SubGroupNumber', noH=True,
                                          physicalUnits=True,
                                          numThreads=8)
            grps = eagle_io.read_array('SUBFIND', eagle_path, snap,
                                       'Subhalo/GroupNumber', noH=True,
                                       physicalUnits=True,
                                       numThreads=8)
            nstars = eagle_io.read_array('SUBFIND', eagle_path, snap,
                                         'Subhalo/SubLengthType', noH=True,
                                         physicalUnits=True,
                                         numThreads=8)[:, 4]

            # Calculate birth redshifts
            zborn = (1 / aborn) - 1

            # Remove particles which are too old
            ages_okinds = zborn < z_100
            pos = pos[ages_okinds, :]
            ini_ms = ini_ms[ages_okinds]
            aborn = aborn[ages_okinds]
            part_subgrp = part_subgrp[ages_okinds]
            part_grp = part_grp[ages_okinds]

            # Create look up dictionary for galaxy values
            d = {"cop": {}, "hmr": {}, "nstar": {}, "m": {},
                 "radii": {}, "ini_ms": {}}
            for (ind, grp), subgrp in zip(enumerate(grps), subgrps):

                # Skip particles not in a galaxy
                if subgrp == 2 ** 30 or nstars[ind] < 100 or gal_ms[ind] < 10 ** 9:
                    continue

                # Store values
                d["cop"][(grp, subgrp)] = cops[ind, :]
                d["hmr"][(grp, subgrp)] = hmrs[ind]
                d["nstar"][(grp, subgrp)] = nstars[ind]
                d["m"][(grp, subgrp)] = gal_ms[ind]
                d["radii"][(grp, subgrp)] = []
                d["ini_ms"][(grp, subgrp)] = []

            # Calculate ages
            ages = calc_ages(z, aborn)

            # Remove particles which are too old
            ages_okinds = ages < 100
            pos = pos[ages_okinds, :]
            ages = ages[ages_okinds]
            ini_ms = ini_ms[ages_okinds]
            aborn = aborn[ages_okinds]
            part_subgrp = part_subgrp[ages_okinds]
            part_grp = part_grp[ages_okinds]

            # Loop over particles calculating and normalising radii
            radii = np.full(ages.size, -1, dtype=np.float64)
            for ind in range(aborn.size):

                # Get grp and subgrp
                grp, subgrp = part_grp[ind], part_subgrp[ind]

                # Get hmr and centre of potential if we are in a valid galaxy
                if (grp, subgrp) in d["hmr"]:
                    hmr = d["hmr"][(grp, subgrp)]
                    cop = d["cop"][(grp, subgrp)]
                    nstar = d["nstar"][(grp, subgrp)]

                    if hmr == 0 or nstar < 100:
                        continue

                    # Centre position
                    this_pos = pos[ind, :] - cop

                    # Compute radius and assign to galaxy entry
                    r = np.sqrt(this_pos[0] ** 2
                                + this_pos[1] ** 2
                                + this_pos[2] ** 2)

                    if r < 30:
                        d["radii"][(grp, subgrp)].append(r)
                        d["ini_ms"][(grp, subgrp)].append(ini_ms[ind])

            # Loop over galaxies calculating profiles
            for key in d["radii"]:

                # Get data
                rs = d["radii"][key]
                this_ini_ms = d["ini_ms"][key]
                gal_m = d["m"][key]
                hmr = d["hmr"][key]

                # Derive radial sfr profile
                binned_stellar_ms, _ = np.histogram(rs,
                                                    bins=radial_bins,
                                                    weights=this_ini_ms)
                radial_sfr = binned_stellar_ms / 100 / gal_m  # 1 / Myr

                # Include this galaxy's profile
                sfr_profile.extend(radial_sfr)
                all_radii.extend(bin_cents / hmr)

            ls = "--"

        else:

            # Get data
            ages = stellar_data[snap]["Particle,S_Age"] * 1000
            ini_ms = stellar_data[snap]["Particle,S_MassInitial"] * 10 ** 10
            radii = stellar_data[snap]["radii"]
            begins = stellar_data[snap]["begin"]
            apps = stellar_data[snap]["Particle/Apertures/Star,30"]
            lengths = stellar_data[snap]["Galaxy,S_Length"]
            hmrs = stellar_data[snap]["HMRs"]
            ms = stellar_data[snap]["Particle,S_Mass"]

            # Create boolean array identifying stars born in the last 100 Myrs
            # and are within the 30 pkpc aperture
            age_okinds = ages < 100
            okinds = np.logical_and(apps, age_okinds)

            # Loop over galaxies normalising by half mass radius
            for igal in range(begins.size):

                # Extract this galaxies data
                b = begins[igal]
                nstar = lengths[igal]
                hmr = hmrs[igal]
                app = apps[b: b + nstar]
                gal_m = np.sum(ms[b: b + nstar][app]) * 10 ** 10

                # Normalise radii
                if hmr <= 0 or gal_m < 10 ** 9:
                    continue

                # Get this galaxy's data
                rs = radii[b: b + nstar][okinds[b: b + nstar]]
                this_ini_ms = ini_ms[b: b + nstar][okinds[b: b + nstar]]

                # Derive radial sfr profile
                binned_stellar_ms, _ = np.histogram(rs,
                                                    bins=radial_bins,
                                                    weights=this_ini_ms)
                radial_sfr = binned_stellar_ms / 100 / gal_m  # 1 / Myr

                # Include this galaxy's profile
                sfr_profile.extend(radial_sfr)
                all_radii.extend(bin_cents / hmr)

            ls = "-"

        # Convert to arrays
        sfr_profile = np.array(sfr_profile)
        all_radii = np.array(all_radii)

        # Plot this profile
        plot_meidan_stat(all_radii, sfr_profile,
                         np.ones(all_radii.size), ax,
                         lab=None, color=cmap(norm(z)),
                         bins=None, ls=ls)

    # Label axes
    ax.set_ylabel("$\mathrm{sSFR}_{100} /[\mathrm{M}_\star /$ Myr]")
    ax.set_xlabel("$R / R_{1/2}$")

    # Create colorbar
    cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm)
    cb.set_label("$z$")

    # Save figure
    mkdir("plots/spatial_dist/")
    fig.savefig("plots/spatial_dist/sfr_radial_profile.png",
                bbox_inches="tight")

    plt.close(fig)
