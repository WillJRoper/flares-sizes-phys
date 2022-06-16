import numpy as np
import h5py
import eagle_IO.eagle_IO as eagle_io
from utils import age2z, calc_3drad, calc_light_mass_rad
from unyt import mh, cm, Gyr, g, Msun, Mpc, kpc
from utils import get_snap_data, clean_data


def compute_stellar_props(stellar_data, snap, path):

    # Define redshift
    z = float(snap.split("z")[-1].replace("p", "."))

    # Define arrays to store computations
    hmrs = np.zeros(len(stellar_data["begin"]))
    mass = np.zeros(len(stellar_data["begin"]))
    w = np.zeros(len(stellar_data["begin"]))
    radii = np.zeros(len(stellar_data["Particle,S_Mass"]))
    den_r = {key: np.zeros(len(stellar_data["begin"]))
             for key in [0.1, 0.5, 1, 30, "hmr"]}
    mass_r = {key: np.zeros(len(stellar_data["begin"]))
              for key in [0.1, 0.5, 1, 30, "hmr"]}
    ages_r = {key: np.zeros(len(stellar_data["begin"]))
              for key in [0.1, 0.5, 1, 30, "hmr"]}
    met_r = {key: np.zeros(len(stellar_data["begin"]))
             for key in [0.1, 0.5, 1, 30, "hmr"]}

    # Loop over galaxies and calculate properties
    for (igal, b), l in zip(enumerate(stellar_data["begin"]),
                            stellar_data["Galaxy,S_Length"]):

        # Get this galaxy's stellar_data
        app = stellar_data["Particle/Apertures/Star,30"][b: b + l]
        cop = stellar_data["Galaxy,COP"][igal]
        pos = stellar_data["Particle,S_Coordinates"][b: b + l, :][app]
        ini_ms = stellar_data["Particle,S_MassInitial"][b: b +
                                                        l][app] * 10 ** 10
        ms = stellar_data["Particle,S_Mass"][b: b + l][app] * 10 ** 10
        ages = stellar_data["Particle,S_Age"][b: b + l][app]
        mets = stellar_data["Particle,S_Z_smooth"][b: b + l][app]

        # Compute particle radii
        rs = calc_3drad(pos - cop)
        radii[b: b + l][app] = rs

        # Compute HMR
        hmr = calc_light_mass_rad(rs, ms, radii_frac=0.5)

        # Compute stellar density within HMR
        if np.sum(ms[rs <= hmr]) > 0:
            den_r["hmr"][igal] = ((np.sum(ms[rs <= hmr])
                                   / (4 / 3 * np.pi * hmr ** 3)) * Msun
                                  / kpc ** 3 / mh).to(1 / cm ** 3).value
            mass_r["hmr"][igal] = np.sum(ms[rs <= hmr])
            ages_r["hmr"][igal] = np.average(ages[rs < hmr],
                                             weights=ini_ms[rs < hmr]) * 10**3
            met_r["hmr"][igal] = np.average(mets[rs < hmr],
                                            weights=ini_ms[rs < hmr])

        # Compute aperture quantities
        for r in [0.1, 0.5, 1, 30]:
            if np.sum(ms[rs <= r]) > 0:
                mass_r[r][igal] = np.sum(ms[rs <= r])
                ages_r[r][igal] = np.average(
                    ages[rs < r], weights=ini_ms[rs < r]) * 10**3
                met_r[r][igal] = np.average(mets[rs < r],
                                            weights=ini_ms[rs < r])
                den_r[r][igal] = (mass_r[r][igal] / (4 / 3 * np.pi * r ** 3)
                                  * Msun / kpc ** 3 / mh).to(1 / cm ** 3).value

        # Store results
        mass[igal] = np.sum(ms)
        hmrs[igal] = hmr
        w[igal] = stellar_data["weights"][igal]

    # Store half mass radii and stellar density within HMR
    stellar_data["HMRs"] = hmrs
    stellar_data["radii"] = radii
    stellar_data["mass"] = mass
    stellar_data["weight"] = w
    stellar_data["apertures"] = {}
    stellar_data["apertures"]["mass"] = mass_r
    stellar_data["apertures"]["age"] = ages_r
    stellar_data["apertures"]["metal"] = met_r
    stellar_data["apertures"]["density"] = den_r

    # Get the index for these star particles
    s_inds = stellar_data["Particle,S_Index"]

    # Get the region for each galaxy
    regions = stellar_data["regions"]

    # Open region overdensities
    reg_ovdens = np.loadtxt("/cosma7/data/dp004/dc-rope1/FLARES/"
                            "flares/region_overdensity.txt",
                            dtype=float)

    # Get the arrays from the raw data files
    aborn = eagle_io.read_array('PARTDATA', path.replace("<reg>", "00"), snap,
                                'PartType4/StellarFormationTime',
                                noH=True, physicalUnits=True,
                                numThreads=8)
    den_born = (eagle_io.read_array("PARTDATA", path.replace("<reg>", "00"),
                                    snap, "PartType4/BirthDensity",
                                    noH=True, physicalUnits=True,
                                    numThreads=8) * 10**10
                * Msun / Mpc ** 3 / mh).to(1 / cm ** 3).value

    # Extract arrays
    zs = np.zeros(s_inds.size)
    dens = np.zeros(s_inds.size)
    w = np.zeros(s_inds.size)
    part_ovdens = np.zeros(s_inds.size)

    # Extract weights for each particle
    prev_reg = 0
    for igal in range(stellar_data["begin"].size):

        # Extract galaxy range
        b = stellar_data["begin"][igal]
        e = b + stellar_data["Galaxy,S_Length"][igal]
        this_s_inds = s_inds[b: e]

        # Get this galaxies region
        reg = regions[igal]

        # Set this galaxy's region overdensity
        part_ovdens[b: e] = reg_ovdens[reg]

        # Set weights for these particles
        w[b: e] = stellar_data["weights"][igal]

        # Open a new region file if necessary
        if reg != prev_reg:
            print("Moving on to region:", reg)

            # Get the arrays from the raw data files
            aborn = eagle_io.read_array('PARTDATA',
                                        path.replace("<reg>",
                                                     str(reg).zfill(2)),
                                        snap,
                                        'PartType4/StellarFormationTime',
                                        noH=True, physicalUnits=True,
                                        numThreads=8)
            den_born = (eagle_io.read_array("PARTDATA",
                                            path.replace("<reg>",
                                                         str(reg).zfill(2)),
                                            snap, "PartType4/BirthDensity",
                                            noH=True, physicalUnits=True,
                                            numThreads=8) * 10**10
                        * Msun / Mpc ** 3 / mh).to(1 / cm ** 3).value
            prev_reg = reg

        # Get this galaxies data
        zs[b: e] = 1 / aborn[this_s_inds] - 1
        dens[b: e] = den_born[this_s_inds]

    # Store the data so we doon't have to recalculate it
    stellar_data["birth_density"] = dens
    stellar_data["birth_z"] = zs
    stellar_data["part_ovdens"] = part_ovdens
    stellar_data["part_weights"] = w

    # Extract arrays
    dens = stellar_data["birth_density"]
    mets = stellar_data["Particle,S_Z_smooth"]
    w = stellar_data["part_weights"]
    radii = stellar_data["radii"]

    # Define how many galaxies we have
    ngal = stellar_data["begin"].size

    # Set up arrays to store results
    gal_bdens = np.zeros(ngal)
    gal_bmet = np.zeros(ngal)
    gal_w = np.zeros(ngal)

    # Loop over each galaxy calculating initial mass weighted properties
    for igal in range(ngal):

        # Extract galaxy range
        b = stellar_data["begin"][igal]
        e = b + stellar_data["Galaxy,S_Length"][igal]
        app = stellar_data["Particle/Apertures/Star,30"][b: e]
        ini_mass = stellar_data["Particle,S_MassInitial"][b: e][app] * 10 ** 10
        hmr = stellar_data["HMRs"][igal]
        rs = radii[b: e][app]
        pdens = dens[b: e][app]
        pmets = mets[b: e][app]

        # Calculate initial mass weighted properties
        okinds = rs != 0
        if np.sum(ini_mass[okinds]) == 0:
            continue
        gal_bdens[igal] = np.average(
            pdens[okinds], weights=ini_mass[okinds])
        gal_bmet[igal] = np.average(
            pmets[okinds], weights=ini_mass[okinds])
        gal_w[igal] = w[b: e][0]

    # Export this data
    stellar_data["gal_birth_density"] = gal_bdens
    stellar_data["gal_birth_Z"] = gal_bmet
    stellar_data["gal_weight"] = gal_w

    return stellar_data


def compute_gas_props(gas_data, stellar_data, snap, path):

    # Define redshift
    z = float(snap.split("z")[-1].replace("p", "."))

    # Define arrays to store computations
    hmrs = np.zeros(len(gas_data["begin"]))
    mass = np.zeros(len(gas_data["begin"]))
    w = np.zeros(len(gas_data["begin"]))
    radii = np.zeros(len(gas_data["Particle,G_Mass"]))
    den_r = {key: np.zeros(len(gas_data["begin"]))
             for key in [0.1, 0.5, 1, 30, "hmr"]}
    mass_r = {key: np.zeros(len(gas_data["begin"]))
              for key in [0.1, 0.5, 1, 30,  "hmr"]}
    met_r = {key: np.zeros(len(gas_data["begin"]))
             for key in [0.1, 0.5, 1, 30, "hmr"]}

    # Loop over galaxies and calculate properties
    for (igal, b), l in zip(enumerate(gas_data["begin"]),
                            gas_data["Galaxy,G_Length"]):

        # Get this galaxy's gas_data
        app = gas_data["Particle/Apertures/Gas,30"][b: b + l]
        cop = gas_data["Galaxy,COP"][igal]
        pos = gas_data["Particle,G_Coordinates"][b: b + l, :][app]
        ms = gas_data["Particle,G_Mass"][b: b + l][app] * 10 ** 10
        mets = gas_data["Particle,G_Z_smooth"][b: b + l][app]

        # Compute particle radii
        rs = calc_3drad(pos - cop)
        radii[b: b + l][app] = rs

        # Compute HMR
        hmr = calc_light_mass_rad(rs, ms, radii_frac=0.5)

        # Compute gas density within HMR
        if np.sum(ms[rs <= hmr]) > 0:
            den_r["hmr"][igal] = ((np.sum(ms[rs <= hmr])
                                   / (4 / 3 * np.pi * hmr ** 3)) * Msun
                                  / kpc ** 3 / mh).to(1 / cm ** 3).value
            mass_r["hmr"][igal] = np.sum(ms[rs <= hmr])
            met_r["hmr"][igal] = np.average(mets[rs < hmr],
                                            weights=ms[rs < hmr])

        # Compute aperture quantities
        for r in [0.1, 0.5, 1, 30]:
            if np.sum(ms[rs <= r]) > 0:
                mass_r[r][igal] = np.sum(ms[rs <= r])
                met_r[r][igal] = np.average(mets[rs < r],
                                            weights=ms[rs < r])
                den_r[r][igal] = (mass_r[r][igal] / (4 / 3 * np.pi * r ** 3)
                                  * Msun / kpc ** 3 / mh).to(1 / cm ** 3).value

        # Store results
        mass[igal] = np.sum(ms)
        hmrs[igal] = hmr
        w[igal] = gas_data["weights"][igal]

    # Store half mass radii and gas density within HMR
    gas_data["HMRs"] = hmrs
    gas_data["radii"] = radii
    gas_data["mass"] = mass
    gas_data["weight"] = w
    gas_data["apertures"] = {}
    gas_data["apertures"]["mass"] = mass_r
    gas_data["apertures"]["metal"] = met_r
    gas_data["apertures"]["density"] = den_r

    # Get the index for these star particles
    g_inds = gas_data["Particle,G_Index"]

    # Get the region for each galaxy
    regions = gas_data["regions"]

    # Open region overdensities
    reg_ovdens = np.loadtxt("/cosma7/data/dp004/dc-rope1/FLARES/"
                            "flares/region_overdensity.txt",
                            dtype=float)

    # Get the arrays from the raw data files
    den_born = (eagle_io.read_array("PARTDATA", path.replace("<reg>", "00"),
                                    snap, "PartType0/Density",
                                    noH=True, physicalUnits=True,
                                    numThreads=8) * 10**10
                * Msun / Mpc ** 3 / mh).to(1 / cm ** 3).value

    # Extract arrays
    dens = np.zeros(g_inds.size)
    w = np.zeros(g_inds.size)
    part_ovdens = np.zeros(g_inds.size)

    # Extract weights for each particle
    prev_reg = 0
    for igal in range(gas_data["begin"].size):

        # Extract galaxy range
        b = gas_data["begin"][igal]
        e = b + gas_data["Galaxy,G_Length"][igal]
        this_s_inds = g_inds[b: e]

        # Get this galaxies region
        reg = regions[igal]

        # Set this galaxy's region overdensity
        part_ovdens[b: e] = reg_ovdens[reg]

        # Set weights for these particles
        w[b: e] = gas_data["weights"][igal]

        # Open a new region file if necessary
        if reg != prev_reg:
            den_born = (eagle_io.read_array("PARTDATA",
                                            path.replace("<reg>",
                                                         str(reg).zfill(2)),
                                            snap, "PartType0/Density",
                                            noH=True, physicalUnits=True,
                                            numThreads=8) * 10**10
                        * Msun / Mpc ** 3 / mh).to(1 / cm ** 3).value
            prev_reg = reg

        # Get this galaxies data
        dens[b: e] = den_born[this_s_inds]

    # Store the data so we doon't have to recalculate it
    gas_data["part_density"] = dens
    gas_data["part_ovdens"] = part_ovdens
    gas_data["part_weights"] = w

    return gas_data


def get_data(snaps, regions, stellar_data_fields, gas_data_fields, path):

    # Define dictionary to hold all data
    data = {"stellar": {s: {} for s in snaps},
            "gas": {s: {} for s in snaps}}

    for snap in snaps:

        # Get the data
        stellar_data = get_snap_data("FLARES", regions, snap,
                                     stellar_data_fields,
                                     length_key="Galaxy,S_Length")
        gas_data = get_snap_data("FLARES", regions, snap, gas_data_fields,
                                 length_key="Galaxy,G_Length")

        # Remove galaxies with fewer than 100 particles
        stellar_data, gas_data = clean_data(stellar_data, gas_data)

        # Compute the derived quantities and store the data
        data["stellar"][snap] = compute_stellar_props(stellar_data, snap, path)
        data["gas"][snap] = compute_gas_props(
            gas_data, stellar_data, snap, path)

    return data
