import numpy as np
from matplotlib.colors import LogNorm

from hmrs import plot_stellar_hmr, plot_stellar_gas_hmr_comp
from density import plot_stellar_density_grid
from stellar_properties import plot_birth_met, plot_birth_den
from stellar_properties import plot_birth_den_vs_met, plot_gal_birth_den_vs_met
from phys_comp import plot_birth_density_evo, plot_birth_met_evo
from phys_comp import plot_hmr_phys_comp, plot_gashmr_phys_comp
from compute_props import get_data


# Define the norm
weight_norm = LogNorm(vmin=10 ** -4, vmax=1)

# Define raw data path
path = "/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/G-EAGLE_<reg>/data/"

# Define regions
regions = []
for reg in range(0, 40):
    if reg < 10:
        regions.append('0' + str(reg))
    else:
        regions.append(str(reg))

# Define FLARES snapshots
# flares_snaps = ['003_z012p000', '004_z011p000', '005_z010p000',
#                 '006_z009p000', '007_z008p000', '008_z007p000',
#                 '009_z006p000', '010_z005p000']
flares_snaps = ['009_z006p000', '010_z005p000']

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

# Get the data we need
data = get_data(flares_snaps, regions, stellar_data_fields, gas_data_fields,
                path)

# Plot the physics variation plots
plot_birth_density_evo()
plot_birth_met_evo()
plot_hmr_phys_comp(flares_snaps[-1])
plot_gashmr_phys_comp(flares_snaps[-1])

# Plot EVERYTHING else
for snap in flares_snaps:

    print("================ Plotting snap %s ================" % snap)

    data["stellar"][snap]["density_cut"] = 10 ** 2.5

    try:
        plot_hmr_phys_comp(snap)
    except ValueError:
        continue

    plot_stellar_hmr(data["stellar"][snap], snap, weight_norm)
    try:
        plot_stellar_density_grid(data["stellar"][snap], snap, weight_norm)
    except ValueError as e:
        print("Stellar density grid:", e)
    plot_stellar_gas_hmr_comp(data["stellar"][snap], data["gas"][snap],
                              snap, weight_norm)

# Plot properties that are done at singular redshifts
snap = flares_snaps[-1]
stellar_data = plot_birth_den(
    data["stellar"][snap], snap, weight_norm, path)
plot_birth_met(data["stellar"][snap], snap, weight_norm, path)
plot_birth_den_vs_met(data["stellar"][snap], snap, weight_norm, path)
stellar_data = plot_gal_birth_den_vs_met(
    data["stellar"][snap], snap, weight_norm, path)

# for snap in eagle_snaps:
#     plot_stellar_hmr("EAGLE", [0, ], snap, weight_norm)
