import numpy as np
import pickle
from matplotlib.colors import LogNorm

from hmrs import *
from density import plot_stellar_density_grid
from stellar_properties import *
from phys_comp import *
from spatial_dist import *
from compute_props import get_data


# Define the norm
weight_norm = LogNorm(vmin=10 ** -4, vmax=1)

# Define raw data path for FLARES and EAGLE
path = "/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/G-EAGLE_<reg>/data/"
agndt9_path = '/cosma7/data/Eagle/ScienceRuns/Planck1/L0050N0752/PE/AGNdT9/data/'
ref_path = "/cosma7/data/Eagle/ScienceRuns/Planck1/L0100N1504/PE/REFERENCE/data"

eagle_path = agndt9_path

# Define regions
regions = []
for reg in range(0, 40):
    if reg < 10:
        regions.append('0' + str(reg))
    else:
        regions.append(str(reg))

# Define FLARES snapshots
flares_snaps = ['005_z010p000', '006_z009p000', '007_z008p000', '008_z007p000',
                '009_z006p000', '010_z005p000']

# Create combined snaps list
com_snaps = ['005_z010p000', '006_z009p000', '007_z008p000', '008_z007p000',
             '009_z006p000', '010_z005p000', '008_z005p037', '010_z003p984',
             '012_z003p017', '015_z002p012']


# Define EAGLE snapshots
pre_snaps = ['000_z020p000', '003_z008p988', '006_z005p971', '009_z004p485',
             '012_z003p017', '015_z002p012', '018_z001p259', '021_z000p736',
             '024_z000p366', '027_z000p101', '001_z015p132', '004_z008p075',
             '007_z005p487', '010_z003p984', '013_z002p478', '016_z001p737',
             '019_z001p004', '022_z000p615', '025_z000p271', '028_z000p000',
             '002_z009p993', '005_z007p050', '008_z005p037', '011_z003p528',
             '014_z002p237', '017_z001p487', '020_z000p865', '023_z000p503',
             '026_z000p183']


# Define data fields
stellar_data_fields = ("Particle,S_Mass", "Particle,S_Coordinates",
                       "Particle/Apertures/Star,1",
                       "Particle/Apertures/Star,5",
                       "Particle/Apertures/Star,10",
                       "Particle/Apertures/Star,30",
                       "Particle,S_Age",
                       "Particle,S_MassInitial",
                       "Particle,S_Z_smooth",
                       "Particle,S_Z",
                       "Particle,S_Index",
                       "Galaxy,COP",
                       "Galaxy,S_Length", "Galaxy,GroupNumber",
                       "Galaxy,SubGroupNumber")

# Define data fields
gas_data_fields = ("Particle,G_Mass", "Particle,G_Coordinates",
                   "Particle,G_Z_smooth", "Particle,G_Index",
                   "Particle/Apertures/Gas,30", "Galaxy,COP",
                   "Galaxy,G_Length", "Galaxy,GroupNumber",
                   "Galaxy,SubGroupNumber")

# Sort EAGLE snapshots
snaps = np.zeros(len(pre_snaps), dtype=object)
for s in pre_snaps:
    ind = int(s.split('_')[0])
    snaps[ind] = s

eagle_snaps = list(snaps)

# PLot the indicative birth density vs metallicity plot
plot_subgrid_birth_den_vs_met()
plot_virial_temp()

# Get the data we need
try:
    with open('data.pck', 'rb') as f:
        data = pickle.load(f)
except OSError:
    data = get_data(flares_snaps, regions, stellar_data_fields, gas_data_fields,
                    path)
    with open('data.pck', 'wb') as f:
        pickle.dump(data, f)

print("Got all data")

# # Make plots that require multiple redshifts
# #sfr_radial_profile_mass(data["stellar"][flares_snaps[-1]], flares_snaps[-1])
# # sfr_radial_profile_environ(data["stellar"][flares_snaps[-1]], flares_snaps[-1])
# sfr_radial_profile(data["stellar"], com_snaps, agndt9_path, flares_snaps)

# print("Plotted SFR profiles")

# # Plot the physics variation plots
# plot_hmr_phys_comp_grid_1kpc(flares_snaps[-1])
plot_potential(flares_snaps[-1])
plot_sfr_evo_comp(flares_snaps[-1])
plot_hmr_phys_comp_grid(flares_snaps[-1])
plot_birth_density_evo()
plot_birth_met_evo()
# plot_hmr_phys_comp(flares_snaps[-1])
# plot_gashmr_phys_comp(flares_snaps[-1])

print("Plotted Physics variations")

# Plot properties that are done at singular redshifts
snap = flares_snaps[-1]
#visualise_gas(data["stellar"][snap], data["gas"][snap], snap, path)
print("Created images")
#plot_sfr_evo(data["stellar"][snap], snap)
plot_birth_met(data["stellar"][snap], snap, weight_norm, path)
plot_birth_den(data["stellar"][snap], snap, weight_norm, path)
plot_eagle_birth_den_vs_met(data["stellar"][snap], snap, weight_norm, path)
plot_gal_birth_den_vs_met(data["stellar"][snap], snap, weight_norm, path)
print("Plotted stellar formation properties")

# Plot EVERYTHING else
for snap in flares_snaps:

    print("================ Plotting snap %s ================" % snap)

    data["stellar"][snap]["density_cut"] = 10 ** 2.0

    try:
        plot_hmr_phys_comp(snap)
    except ValueError:
        continue

    plot_stellar_hmr(data["stellar"][snap], snap, weight_norm)
    plot_stellar_gas_hmr_comp(data["stellar"][snap], data["gas"][snap],
                              snap, weight_norm)

# for snap in eagle_snaps:
#     plot_stellar_hmr("EAGLE", [0, ], snap, weight_norm)
