import numpy as np
import matplotlib as mpl
import matplotlib.colors as cm
import matplotlib.pyplot as plt
from flare import plt as flareplt
from utils import mkdir, plot_meidan_stat, calc_ages
import eagle_IO.eagle_IO as eagle_io


def sfr_radial_profile(stellar_data, snaps, eagle_path):

    # Define radial bins
    radial_bins = np.logspace(-2, 10, 50)
    bin_cents = (radial_bins[:-1] + radial_bins[1:]) / 2

    # Define redshift colormap and normalisation
    cmap = mpl.cm.get_cmap('plasma', len(snaps))
    norm = cm.Normalize(vmin=0, vmax=10)

    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Loop over snapshots
    for snap in snaps:

        print("Plotting sfr radial profile for snap", snap)

        # Get redshift
        z = float(snap.split("z")[-1].replace("p", "."))

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
                                       numThreads=8) * 10**3
            subgrp = eagle_io.read_array('SUBFIND', eagle_path, snap,
                                         'Subhalo/SubGroupNumber', noH=True,
                                         physicalUnits=True,
                                         numThreads=8)
            grp = eagle_io.read_array('SUBFIND', eagle_path, snap,
                                      'Subhalo/GroupNumber', noH=True,
                                      physicalUnits=True,
                                      numThreads=8)

            # Create look up dictionary for galaxy values
            d = {"cop": {}, "hmr": {}}
            for ind, (grp, subgrp) in enumerate(zip(grp. subgrp)):

                # Skip particles not in a galaxy
                if subgrp == 2 ** 30:
                    continue

                # Store values
                d["cop"][(grp, subgrp)] = cops[ind]
                d["hmr"][(grp, subgrp)] = hmrs[ind]

            # Calculate ages
            ages = calc_ages(z, aborn)

            # Loop over particles calculating and normalising radii
            radii = np.full(aborn.size, -1)
            for ind in range(aborn.size):

                # Get grp and subgrp
                grp, subgrp = part_grp[ind], part_subgrp[ind]

                # Get hmr and centre of potential if we are in a valid galaxy
                if (grp, subgrp) in d["hmr"]:
                    hmr = d["hmr"][(grp, subgrp)]
                    cop = d["cop"][(grp, subgrp)]

                    # Centre position
                    this_pos = pos[ind, :] - cop

                    # Compute and normalise  radius
                    radii[ind] = np.sqrt(this_pos[0] ** 2
                                         + this_pos[1] ** 2
                                         + this_pos[2] ** 2) / hmr

            # Define boolean arrays for age and the 30 pkpc aperture
            age_okinds = np.logical_and(ages < 100)
            okinds = np.logical_and(radii <= 30, age_okinds)

        else:

            # Get data
            ages = stellar_data[snap]["Particle,S_Age"] * 1000
            ini_ms = stellar_data[snap]["Particle,S_MassInitial"] * 10 ** 10
            radii = stellar_data[snap]["radii"]
            begins = stellar_data[snap]["begin"]
            apps = stellar_data[snap]["Particle/Apertures/Star,30"]
            lengths = stellar_data[snap]["Galaxy,S_Length"]
            hmrs = stellar_data[snap]["HMRs"]

            # Create boolean array identifying stars born in the last 100 Myrs
            # and are within the 30 pkpc aperture
            age_okinds = np.logical_and(ages < 100)
            okinds = np.logical_and(apps, age_okinds)

            # Loop over galaxies normalising by half mass radius
            for igal in range(begins.size):

                # Extract this galaxies data
                b = begins[igal]
                nstar = lengths[igal]
                hmr = hmrs[igal]

                # Normalise radii
                if hmr > 0:
                    radii[b: b + nstar] /= hmr
                else:
                    radii[b: b + nstar] = -1

        # Remove anomalous galaxies (with hmr = 0)
        okinds = np.logical_and(okinds, radii >= 0)

        # Derive radial sfr profile
        binned_stellar_ms, _ = np.histogram(radii[okinds], bins=radial_bins,
                                            weights=ini_ms[okinds])
        radial_sfr = binned_stellar_ms / 100  # M_sun / Myr

        # Plot this profile
        ax.plot(bin_cents, radial_sfr, color=cmap(norm(z)))

    # Label axes
    ax.set_ylabel("$\mathrm{SFR}_{100} /$ [M_\star / Myr]")
    ax.set_xlabel("$R / R_{\star, 1/2}$")

    # Create colorbar
    cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm)
    cb.set_label("$z$")

    # Save figure
    mkdir("plots/spatial_dist/")
    fig.savefig("plots/spatial_dist/sfr_radial_profile.png",
                bbox_inches="tight")

    plt.close(fig)
