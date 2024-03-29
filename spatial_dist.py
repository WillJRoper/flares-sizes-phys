import numpy as np
import matplotlib as mpl
import matplotlib.colors as cm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from flare import plt as flareplt
from utils import mkdir, plot_meidan_stat, calc_ages, plot_spread_stat_as_eb
from utils import weighted_quantile
from astropy.cosmology import Planck18 as cosmo
from astropy.cosmology import z_at_value
import astropy.units as u
import eagle_IO.eagle_IO as eagle_io
from scipy.optimize import curve_fit


def sfr_radial_profile(stellar_data, snaps, agndt9_path, flares_snaps):

    # Define radial bins
    radial_bins = np.logspace(-2, 1.8, 50)
    bin_cents = (radial_bins[:-1] + radial_bins[1:]) / 2

    # Define redshift colormap and normalisation
    cmap = mpl.cm.get_cmap('plasma')
    norm = cm.Normalize(vmin=2, vmax=10)

    # Define REF path that's only used here
    ref_path = "/cosma7/data/Eagle/ScienceRuns/Planck1/L0100N1504/PE/REFERENCE/data"

    # Initialise legend elements
    legend_elements = []

    # Set up plot
    fig = plt.figure(figsize=(3.5, 3.5))
    gs = gridspec.GridSpec(nrows=1, ncols=1 + 1,
                           width_ratios=[20, ] + [1, ])
    gs.update(wspace=0.0, hspace=0.0)
    ax = fig.add_subplot(gs[0, 0])
    ax.loglog()
    cax = fig.add_subplot(gs[0, 1])

    # Loop over snapshots
    for snap in snaps:

        print("Plotting sfr radial profile for snap", snap)

        # Get redshift
        z = float(snap.split("z")[-1].replace("p", "."))

        # Calculate redshift 100 Myrs before this z
        z_100 = z_at_value(cosmo.age, cosmo.age(z) - 100 * u.Myr)

        # Are we dealing with EAGLE or FLARES?
        if not snap in flares_snaps:

            for eagle_path in [ref_path, agndt9_path]:

                if eagle_path == agndt9_path and snap == '008_z005p037':
                    continue

                # Initialise lists to hold the sfr profiles
                sfr_profile = []
                all_radii = []

                aborn = eagle_io.read_array('PARTDATA', eagle_path, snap,
                                            'PartType4/StellarFormationTime',
                                            noH=True,
                                            physicalUnits=True,
                                            numThreads=8)
                ini_ms = eagle_io.read_array('PARTDATA', eagle_path, snap,
                                             'PartType4/InitialMass', noH=True,
                                             physicalUnits=True,
                                             numThreads=8) * 10 ** 10
                ms = eagle_io.read_array('PARTDATA', eagle_path, snap,
                                         'PartType4/Mass', noH=True,
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

                # Create look up dictionary for galaxy values
                d = {"cop": {}, "hmr": {}, "nstar": {}, "m": {},
                     "all_radii": {}, "radii": {}, "ini_ms": {}, "ms": {}}
                for (ind, grp), subgrp in zip(enumerate(grps), subgrps):

                    # Skip particles not in a galaxy
                    if nstars[ind] < 100 or gal_ms[ind] < 10 ** 9:
                        continue

                    # Store values
                    d["cop"][(grp, subgrp)] = cops[ind, :]
                    d["hmr"][(grp, subgrp)] = hmrs[ind]
                    d["nstar"][(grp, subgrp)] = nstars[ind]
                    d["m"][(grp, subgrp)] = gal_ms[ind]
                    d["radii"][(grp, subgrp)] = []
                    d["all_radii"][(grp, subgrp)] = []
                    d["ini_ms"][(grp, subgrp)] = []
                    d["ms"][(grp, subgrp)] = []

                # Loop over particles born recently
                for ind in range(aborn.size):

                    # Get grp and subgrp
                    grp, subgrp = part_grp[ages_okinds][ind], part_subgrp[ages_okinds][ind]

                    # Get hmr and centre of potential if we are
                    # in a valid galaxy
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

                # Loop over particles all particles
                for ind in range(ms.size):

                    # Get grp and subgrp
                    grp, subgrp = part_grp[ind], part_subgrp[ind]

                    # Get hmr and centre of potential if we are
                    # in a valid galaxy
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
                            d["all_radii"][(grp, subgrp)].append(r)
                            d["ms"][(grp, subgrp)].append(ms[ind])

                # Loop over galaxies calculating profiles
                for key in d["radii"]:

                    # Get data
                    rs = d["radii"][key]
                    all_rs = d["all_radii"][key]
                    this_ini_ms = d["ini_ms"][key]
                    this_ms = d["ms"][key]
                    gal_m = d["m"][key]

                    if len(this_ini_ms) == 0:
                        continue

                    # Derive radial sfr profile
                    binned_stellar_ini_ms, _ = np.histogram(rs,
                                                            bins=radial_bins,
                                                            weights=this_ini_ms)
                    binned_stellar_ms, _ = np.histogram(all_rs,
                                                        bins=radial_bins,
                                                        weights=this_ms)
                    radial_sfr = binned_stellar_ini_ms / 100 / binned_stellar_ms  # 1 / Myr

                    # Include this galaxy's profile
                    sfr_profile.extend(radial_sfr)
                    all_radii.extend(bin_cents)

                # Assign linestyle
                if eagle_path == ref_path:
                    ls = "dotted"
                    marker = "^"
                else:
                    ls = "--"
                    marker = "s"

                # Convert to arrays
                sfr_profile = np.array(sfr_profile)
                all_radii = np.array(all_radii)

                # Plot this profile
                plot_meidan_stat(all_radii, sfr_profile,
                                 np.ones(all_radii.size), ax,
                                 lab=None, color=cmap(norm(z)),
                                 bins=radial_bins, ls=ls)

        else:

            # Initialise lists to hold the sfr profiles
            sfr_profile = []
            all_radii = []
            all_ws = []

            # Get data
            ages = stellar_data[snap]["Particle,S_Age"][...] * 1000
            ini_ms = stellar_data[snap]["Particle,S_MassInitial"][...] * 10 ** 10
            radii = stellar_data[snap]["radii"][...]
            begins = stellar_data[snap]["begin"][...]
            apps = stellar_data[snap]["Particle/Apertures/Star,30"][...]
            lengths = stellar_data[snap]["Galaxy,S_Length"][...]
            hmrs = stellar_data[snap]["HMRs"][...]
            ms = stellar_data[snap]["Particle,S_Mass"][...]
            w = stellar_data[snap]["weights"][...]

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
                ws = w[igal]
                nstar_100 = ini_ms[b: b + nstar][okinds[b: b + nstar]].size

                # Normalise radii
                if hmr <= 0 or gal_m < 10 ** 9 or nstar_100 == 0:
                    continue

                # Get this galaxy's data
                rs = radii[b: b + nstar][okinds[b: b + nstar]]
                all_rs = radii[b: b + nstar][app]
                this_ini_ms = ini_ms[b: b + nstar][okinds[b: b + nstar]]
                this_ms = ms[b: b + nstar][app] * 10 ** 10

                # Derive radial sfr profile
                binned_stellar_ini_ms, _ = np.histogram(rs,
                                                        bins=radial_bins,
                                                        weights=this_ini_ms)
                binned_stellar_ms, _ = np.histogram(all_rs,
                                                    bins=radial_bins,
                                                    weights=this_ms)
                radial_sfr = binned_stellar_ini_ms / 100 / binned_stellar_ms  # 1 / Myr

                # Include this galaxy's profile
                sfr_profile.extend(radial_sfr)
                all_radii.extend(bin_cents)
                all_ws.extend(np.full(bin_cents.size, ws))

            # Convert to arrays
            sfr_profile = np.array(sfr_profile)
            all_radii = np.array(all_radii)
            all_ws = np.array(all_ws)

            # Plot this profile
            plot_meidan_stat(all_radii, sfr_profile,
                             all_ws, ax,
                             lab=None, color=cmap(norm(z)),
                             bins=radial_bins, ls="-")

    # Set up legend
    legend_elements.append(Line2D([0], [0], color='k',
                                  label="FLARES",
                                  linestyle="-"))
    legend_elements.append(Line2D([0], [0], color='k',
                                  label="EAGLE-AGNdT9",
                                  linestyle="--"))
    legend_elements.append(Line2D([0], [0], color='k',
                                  label="EAGLE-REF",
                                  linestyle="dotted"))
    ax.legend(handles=legend_elements,
              loc='upper center',
              bbox_to_anchor=(0.5, -0.25),
              fancybox=True, ncol=3)

    # Label axes
    ax.set_ylabel("$\mathrm{sSFR}_{100} / [\mathrm{Myr}^{-1}]$")
    ax.set_xlabel("$R / $[pkpc]")

    # Create colorbar
    cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm)
    cb.set_label("$z$")

    # Save figure
    mkdir("plots/spatial_dist/")
    fig.savefig("plots/spatial_dist/sfr_radial_profile.png",
                bbox_inches="tight")

    plt.close(fig)


def sfr_radial_profile_environ(stellar_data, snap):

    # Define radial bins
    radial_bins = np.logspace(-2, 1.8, 50)
    bin_cents = (radial_bins[:-1] + radial_bins[1:]) / 2

    # Define overdensity bins in log(1+delta)
    ovden_bins = np.arange(-0.3, 0.4, 0.1)

    # Define redshift colormap and normalisation
    cmap = mpl.cm.get_cmap('magma', len(ovden_bins))
    norm = cm.Normalize(vmin=ovden_bins.min(), vmax=ovden_bins.max())

    # Set up plot
    fig = plt.figure(figsize=(3.5, 3.5))
    gs = gridspec.GridSpec(nrows=1, ncols=1 + 1,
                           width_ratios=[20, ] + [1, ])
    gs.update(wspace=0.0, hspace=0.0)
    ax = fig.add_subplot(gs[0, 0])
    ax.loglog()
    cax = fig.add_subplot(gs[0, 1])

    # Get redshift
    z = float(snap.split("z")[-1].replace("p", "."))

    # Initialise lists to hold the sfr profiles
    sfr_profile = []
    all_radii = []
    all_ws = []
    all_ovdens = []

    # Get data
    ages = stellar_data["Particle,S_Age"] * 1000
    ini_ms = stellar_data["Particle,S_MassInitial"] * 10 ** 10
    radii = stellar_data["radii"]
    begins = stellar_data["begin"]
    apps = stellar_data["Particle/Apertures/Star,30"]
    lengths = stellar_data["Galaxy,S_Length"]
    hmrs = stellar_data["HMRs"]
    ms = stellar_data["Particle,S_Mass"]
    w = stellar_data["weights"]
    part_ovdens = stellar_data["part_ovdens"]

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
        nstar_100 = ini_ms[b: b + nstar][okinds[b: b + nstar]].size

        # Normalise radii
        if hmr <= 0 or gal_m < 10 ** 9 or nstar_100 == 0:
            continue

        # Get this galaxy's data
        rs = radii[b: b + nstar][okinds[b: b + nstar]]
        this_ini_ms = ini_ms[b: b + nstar][okinds[b: b + nstar]]
        this_w = w[igal]
        ovden = part_ovdens[b: b + nstar][okinds[b: b + nstar]][0]

        # Derive radial sfr profile
        binned_stellar_ms, _ = np.histogram(rs,
                                            bins=radial_bins,
                                            weights=this_ini_ms)
        radial_sfr = binned_stellar_ms / 100 / gal_m  # 1 / Myr

        # Include this galaxy's profile
        sfr_profile.extend(radial_sfr)
        all_radii.extend(bin_cents)
        all_ws.extend(np.full(bin_cents.size, this_w))
        all_ovdens.extend(np.full(bin_cents.size, ovden))

    # Convert to arrays
    sfr_profile = np.array(sfr_profile)
    all_radii = np.array(all_radii)
    all_ovdens = np.array(all_ovdens)
    all_ws = np.array(all_ws)

    # Loop over overdensity bins and plot median curves
    for i in range(ovden_bins[:-1].size):

        bin_cent = (ovden_bins[i + 1] + ovden_bins[i]) / 2

        # Get boolean indices for this bin
        okinds = np.logical_and(all_ovdens < ovden_bins[i + 1],
                                all_ovdens >= ovden_bins[i])

        # Plot this profile
        plot_meidan_stat(all_radii[okinds], sfr_profile[okinds],
                         np.ones(all_ws[okinds].size), ax,
                         lab=None, color=cmap(norm(bin_cent)),
                         bins=radial_bins, ls="-")

    # Label axes
    ax.set_ylabel("$\mathrm{sSFR}_{100} / [\mathrm{Myr}^{-1}]$")
    ax.set_xlabel("$R / $[pkpc]")

    # Create colorbar
    cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm)
    cb.set_label("$\log_{10}(1 + \delta)$")

    # Save figure
    mkdir("plots/spatial_dist/")
    fig.savefig("plots/spatial_dist/sfr_radial_profile_environ.png",
                bbox_inches="tight")

    plt.close(fig)


def sfr_radial_profile_mass(stellar_data, snap):

    # Define radial bins
    radial_bins = np.logspace(-2, 1.8, 50)
    bin_cents = (radial_bins[:-1] + radial_bins[1:]) / 2

    # Define overdensity bins in log(1+delta)
    mass_bins = np.logspace(9, 11.6, 5)

    # Define redshift colormap and normalisation
    cmap = mpl.cm.get_cmap('magma')
    norm = cm.LogNorm(vmin=mass_bins.min(), vmax=mass_bins.max())

    # Set up plot
    fig = plt.figure(figsize=(3.5, 3.5))
    gs = gridspec.GridSpec(nrows=1, ncols=1 + 1,
                           width_ratios=[20, ] + [1, ])
    gs.update(wspace=0.0, hspace=0.0)
    ax = fig.add_subplot(gs[0, 0])
    ax.loglog()
    cax = fig.add_subplot(gs[0, 1])

    # Get redshift
    z = float(snap.split("z")[-1].replace("p", "."))

    # Initialise lists to hold the sfr profiles
    sfr_profile = []
    all_radii = []
    all_ws = []
    all_gal_ms = []

    # Get data
    ages = stellar_data["Particle,S_Age"] * 1000
    ini_ms = stellar_data["Particle,S_MassInitial"] * 10 ** 10
    radii = stellar_data["radii"]
    begins = stellar_data["begin"]
    apps = stellar_data["Particle/Apertures/Star,30"]
    lengths = stellar_data["Galaxy,S_Length"]
    hmrs = stellar_data["HMRs"]
    ms = stellar_data["Particle,S_Mass"]
    w = stellar_data["weights"]

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
        nstar_100 = ini_ms[b: b + nstar][okinds[b: b + nstar]].size

        # Normalise radii
        if hmr <= 0 or gal_m < 10 ** 9 or nstar_100 == 0:
            continue

        # Get this galaxy's data
        rs = radii[b: b + nstar][okinds[b: b + nstar]]
        this_ini_ms = ini_ms[b: b + nstar][okinds[b: b + nstar]]
        this_w = w[igal]

        # Derive radial sfr profile
        binned_stellar_ms, _ = np.histogram(rs,
                                            bins=radial_bins,
                                            weights=this_ini_ms)
        radial_sfr = binned_stellar_ms / 100 / gal_m  # 1 / Myr

        # Include this galaxy's profile
        sfr_profile.extend(radial_sfr)
        all_radii.extend(bin_cents)
        all_ws.extend(np.full(bin_cents.size, this_w))
        all_gal_ms.extend(np.full(bin_cents.size, gal_m))

    # Convert to arrays
    sfr_profile = np.array(sfr_profile)
    all_radii = np.array(all_radii)
    all_gal_ms = np.array(all_gal_ms)
    all_ws = np.array(all_ws)

    # Loop over overdensity bins and plot median curves
    for i in range(mass_bins[:-1].size):

        bin_cent = (mass_bins[i + 1] + mass_bins[i]) / 2

        # Get boolean indices for this bin
        okinds = np.logical_and(all_gal_ms < mass_bins[i + 1],
                                all_gal_ms >= mass_bins[i])

        # Plot this profile
        plot_meidan_stat(all_radii[okinds], sfr_profile[okinds],
                         all_ws[okinds], ax,
                         lab=None, color=cmap(norm(bin_cent)),
                         bins=radial_bins, ls="-")

    # Label axes
    ax.set_ylabel("$\mathrm{sSFR}_{100} / [\mathrm{Myr}^{-1}]$")
    ax.set_xlabel("$R / $[pkpc]")

    # Create colorbar
    cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm)
    cb.set_label("$M_\star / M_\odot$")

    # Save figure
    mkdir("plots/spatial_dist/")
    fig.savefig("plots/spatial_dist/sfr_radial_profile_mass.png",
                bbox_inches="tight")

    plt.close(fig)


def plot_dead_inside(stellar_data, snaps, weight_norm):

    for snap in snaps:

        print(snap)

        # Initialise lists to hold the sfr profiles
        grads = []
        star_ms = []
        all_ws = []
        in_ssfrs = []
        out_ssfrs = []
        ssfrs = []

        # Get data
        ages = stellar_data[snap]["Particle,S_Age"][...] * 1000
        ini_ms = stellar_data[snap]["Particle,S_MassInitial"][...] * 10 ** 10
        radii = stellar_data[snap]["radii"][...]
        begins = stellar_data[snap]["begin"][...]
        apps = stellar_data[snap]["Particle/Apertures/Star,30"][...]
        lengths = stellar_data[snap]["Galaxy,S_Length"][...]
        hmrs = stellar_data[snap]["HMRs"][...]
        ms = stellar_data[snap]["Particle,S_Mass"][...]
        w = stellar_data[snap]["weights"][...]

        # Create boolean array identifying stars born in the last 100 Myrs
        # and are within the 30 pkpc aperture
        age_okinds = ages < 100
        okinds = np.logical_and(apps, age_okinds)

        # Define straight line fit
        def sline(x, m, c): return m * x + c

        # Loop over galaxies normalising by half mass radius
        for igal in range(begins.size):

            # Extract this galaxies data
            b = begins[igal]
            nstar = lengths[igal]
            hmr = hmrs[igal]
            app = apps[b: b + nstar]
            gal_m = np.sum(ms[b: b + nstar][app]) * 10 ** 10
            nstar_100 = ini_ms[b: b + nstar][okinds[b: b + nstar]].size

            # Ignore anomalous galaxies and low mass galaxies
            if hmr <= 0 or nstar_100 < 100:
                continue

            # Get this galaxy's data
            rs = radii[b: b + nstar][okinds[b: b + nstar]]
            all_rs = radii[b: b + nstar][app]
            this_ini_ms = ini_ms[b: b + nstar][okinds[b: b + nstar]]
            this_ms = ms[b: b + nstar][app] * 10 ** 10

            r_lim = np.median(all_rs)

            in_ini_ms = this_ini_ms[rs <= r_lim]
            in_ms = this_ms[all_rs <= r_lim]
            out_ini_ms = this_ini_ms[rs > r_lim]
            out_ms = this_ms[all_rs > r_lim]

            in_ssfr = np.sum(in_ini_ms) / 100 / gal_m  # 1 / Myr
            out_ssfr = np.sum(out_ini_ms) / 100 / gal_m  # 1 / Myr
            ssfr = np.sum(this_ini_ms) / 100 / gal_m

            # Fit the radial profile
            grad = in_ssfr / out_ssfr

            # Include this galaxy's profile
            grads.append(grad)
            star_ms.append(gal_m)
            all_ws.append(w[igal])
            in_ssfrs.append(in_ssfr)
            out_ssfrs.append(out_ssfr)
            ssfrs.append(ssfr)

        # Convert to arrays
        grads = np.array(grads)
        star_ms = np.array(star_ms)
        all_ws = np.array(all_ws)
        in_ssfrs = np.array(in_ssfrs)
        out_ssfrs = np.array(out_ssfrs)
        ssfrs = np.array(ssfrs)

        # Set up plot
        fig = plt.figure(figsize=(3.5, 3.5))
        gs = gridspec.GridSpec(nrows=1, ncols=1 + 1,
                               width_ratios=[20, ] + [1, ])
        gs.update(wspace=0.0, hspace=0.0)
        ax = fig.add_subplot(gs[0, 0])
        cax = fig.add_subplot(gs[0, 1])

        # Remove zeros
        okinds = grads > 0

        # Plot stellar_data
        im = ax.hexbin(star_ms[okinds], grads[okinds], gridsize=30,
                       mincnt=np.min(all_ws) - (0.1 * np.min(all_ws)),
                       C=all_ws[okinds],
                       reduce_C_function=np.sum, xscale='log', yscale="log",
                       norm=weight_norm, linewidths=0.2, cmap='viridis')

        # Label axes
        ax.set_xlabel("$M_\star / M_\odot$")
        ax.set_ylabel(
            "$\mathrm{sSFR}_{\mathrm{in}} / \mathrm{sSFR}_{\mathrm{out}}$")

        # Create colorbar
        cb = fig.colorbar(im, cax)
        cb.set_label("$\sum w_{i}$")

        # Save figure
        mkdir("plots/spatial_dist/")
        fig.savefig("plots/spatial_dist/dead_inside_grad_%s.png" % snap,
                    bbox_inches="tight")

        plt.close(fig)

        # Set up plot
        fig = plt.figure(figsize=(3.5, 3.5))
        gs = gridspec.GridSpec(nrows=1, ncols=1 + 1,
                               width_ratios=[20, ] + [1, ])
        gs.update(wspace=0.0, hspace=0.0)
        ax = fig.add_subplot(gs[0, 0])
        cax = fig.add_subplot(gs[0, 1])

        # Plot stellar_data
        im = ax.hexbin(in_ssfrs, out_ssfrs, gridsize=30,
                       mincnt=np.min(all_ws) - (0.1 * np.min(all_ws)),
                       C=star_ms, norm=cm.LogNorm(),
                       reduce_C_function=np.mean, linewidths=0.2, cmap='viridis')

        # Get the limits for 1-1 reltion
        low = np.min((ax.get_xlim()[0], ax.get_ylim()[0]))
        up = np.max((ax.get_xlim()[1], ax.get_ylim()[1]))

        ax.plot([low, up], [low, up],
                color="k", linestyle="--", alpha=0.8)

        # Label axes
        ax.set_xlabel(
            "$\mathrm{sSFR}^{\mathrm{in}}_{100} / [\mathrm{Myr}^{-1}]$")
        ax.set_ylabel(
            "$\mathrm{sSFR}^{\mathrm{out}}_{100} / [\mathrm{Myr}^{-1}]$")

        # Create colorbar
        cb = fig.colorbar(im, cax)
        cb.set_label("$M_\star / M_\odot$")

        # Save figure
        mkdir("plots/spatial_dist/")
        fig.savefig("plots/spatial_dist/dead_inside_ssfrs_%s.png" % snap,
                    bbox_inches="tight")

        plt.close(fig)

        # Set up plot
        fig = plt.figure(figsize=(3.5, 3.5))
        gs = gridspec.GridSpec(nrows=1, ncols=1 + 1,
                               width_ratios=[20, ] + [1, ])
        gs.update(wspace=0.0, hspace=0.0)
        ax = fig.add_subplot(gs[0, 0])
        cax = fig.add_subplot(gs[0, 1])

        # Plot stellar_data
        okinds = np.logical_and(in_ssfrs > 0, out_ssfrs > 0)
        im = ax.hexbin(in_ssfrs[okinds], out_ssfrs[okinds], gridsize=30,
                       mincnt=0,
                       C=star_ms[okinds], norm=cm.LogNorm(), xscale="log", yscale="log",
                       reduce_C_function=np.mean, linewidths=0.2, cmap='viridis')

        # Get the limits for 1-1 reltion
        low = np.min((ax.get_xlim()[0], ax.get_ylim()[0]))
        up = np.max((ax.get_xlim()[1], ax.get_ylim()[1]))

        ax.plot([low, up], [low, up],
                color="k", linestyle="--", alpha=0.8)

        # Label axes
        ax.set_xlabel(
            "$\mathrm{sSFR}^{\mathrm{in}}_{100} / [\mathrm{Myr}^{-1}]$")
        ax.set_ylabel(
            "$\mathrm{sSFR}^{\mathrm{out}}_{100} / [\mathrm{Myr}^{-1}]$")

        # Create colorbar
        cb = fig.colorbar(im, cax)
        cb.set_label("$M_\star / M_\odot$")

        # Save figure
        mkdir("plots/spatial_dist/")
        fig.savefig("plots/spatial_dist/dead_inside_ssfrs_logged%s.png" % snap,
                    bbox_inches="tight")

        plt.close(fig)

        # Set up plot
        fig = plt.figure(figsize=(3.5, 3.5))
        gs = gridspec.GridSpec(nrows=1, ncols=1 + 1,
                               width_ratios=[20, ] + [1, ])
        gs.update(wspace=0.0, hspace=0.0)
        ax = fig.add_subplot(gs[0, 0])
        cax = fig.add_subplot(gs[0, 1])

        # Plot stellar_data
        okinds = np.logical_and(ssfrs > 0, out_ssfrs > 0)
        im = ax.hexbin(ssfrs[okinds], out_ssfrs[okinds], gridsize=30,
                       mincnt=0,
                       C=star_ms[okinds], norm=cm.LogNorm(), xscale="log", yscale="log",
                       reduce_C_function=np.mean, linewidths=0.2, cmap='viridis')

        # Label axes
        ax.set_xlabel(
            "$\mathrm{sSFR}_{100} / [\mathrm{Myr}^{-1}]$")
        ax.set_ylabel(
            "$\mathrm{sSFR}^{\mathrm{out}}_{100} / [\mathrm{Myr}^{-1}]$")

        # Create colorbar
        cb = fig.colorbar(im, cax)
        cb.set_label("$M_\star / M_\odot$")

        # Save figure
        mkdir("plots/spatial_dist/")
        fig.savefig("plots/spatial_dist/dead_outside_vsssfr%s.png" % snap,
                    bbox_inches="tight")

        plt.close(fig)

        # Set up plot
        fig = plt.figure(figsize=(3.5, 3.5))
        gs = gridspec.GridSpec(nrows=1, ncols=1 + 1,
                               width_ratios=[20, ] + [1, ])
        gs.update(wspace=0.0, hspace=0.0)
        ax = fig.add_subplot(gs[0, 0])
        cax = fig.add_subplot(gs[0, 1])

        # Plot stellar_data
        okinds = np.logical_and(ssfrs > 0, in_ssfrs > 0)
        im = ax.hexbin(ssfrs[okinds], in_ssfrs[okinds], gridsize=30,
                       mincnt=0,
                       C=star_ms[okinds], norm=cm.LogNorm(), xscale="log", yscale="log",
                       reduce_C_function=np.mean, linewidths=0.2, cmap='viridis')

        # Label axes
        ax.set_xlabel(
            "$\mathrm{sSFR}_{100} / [\mathrm{Myr}^{-1}]$")
        ax.set_ylabel(
            "$\mathrm{sSFR}^{\mathrm{in}}_{100} / [\mathrm{Myr}^{-1}]$")

        # Create colorbar
        cb = fig.colorbar(im, cax)
        cb.set_label("$M_\star / M_\odot$")

        # Save figure
        mkdir("plots/spatial_dist/")
        fig.savefig("plots/spatial_dist/dead_inside_vsssfr%s.png" % snap,
                    bbox_inches="tight")

        plt.close(fig)
