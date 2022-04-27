import numpy as np
from matplotlib.colors import LogNorm

from hmrs import plot_stellar_hmr, plot_stellar_gas_hmr_comp
from density import plot_stellar_density
from utils import get_data

# Define the norm
weight_norm = LogNorm(vmin=10 ** -4, vmax=1)

# Define regions
regions = []
for reg in range(0, 40):
    if reg < 10:
        regions.append('0' + str(reg))
    else:
        regions.append(str(reg))

# Define FLARES snapshots
flares_snaps = ['000_z015p000', '001_z014p000', '002_z013p000',
                '003_z012p000', '004_z011p000', '005_z010p000',
                '006_z009p000', '007_z008p000', '008_z007p000',
                '009_z006p000', '010_z005p000']

# Define EAGLE snapshots
pre_snaps = ['000_z020p000', '003_z008p988', '006_z005p971', '009_z004p485',
             '012_z003p017', '015_z002p012', '018_z001p259', '021_z000p736',
             '024_z000p366', '027_z000p101', '001_z015p132', '004_z008p075',
             '007_z005p487', '010_z003p984', '013_z002p478', '016_z001p737',
             '019_z001p004', '022_z000p615', '025_z000p271', '028_z000p000',
             '002_z009p993', '005_z007p050', '008_z005p037', '011_z003p528',
             '014_z002p237', '017_z001p487', '020_z000p865', '023_z000p503',
             '026_z000p183']

# Sort EAGLE snapshots
snaps = np.zeros(len(pre_snaps), dtype=object)
for s in pre_snaps:
    ind = int(s.split('_')[0])
    snaps[ind] = s

eagle_snaps = list(snaps)

# Plot EVERYTHING
for snap in flares_snaps:

    # Define data fields
    stellar_data_fields = ("Particle,S_Mass", "Particle,S_Coordinates",
                           "Particle/Apertures/Star,1",
                           "Particle/Apertures/Star,5",
                           "Particle/Apertures/Star,10",
                           "Particle/Apertures/Star,30", "Galaxy,COP",
                           "Galaxy,S_Length", "Galaxy,GroupNumber",
                           "Galaxy,SubGroupNumber")

    # Define data fields
    gas_data_fields = ("Particle,G_Mass", "Particle,G_Coordinates",
                       "Particle/Apertures/Gas,30", "Galaxy,COP",
                       "Galaxy,G_Length", "Galaxy,GroupNumber",
                       "Galaxy,SubGroupNumber")

    # Get the data
    stellar_data = get_data("FLARES", regions, snap, stellar_data_fields,
                            length_key="Galaxy,S_Length")
    gas_data = get_data("FLARES", regions, snap, gas_data_fields,
                        length_key="Galaxy,G_Length")

    print("================ Plotting snap %s ================" % snap)
    try:
        plot_stellar_density(stellar_data, snap, weight_norm)
    except ValueError as e:
        print("Stellar density:", e)
    plot_stellar_hmr(stellar_data, snap, weight_norm)
    plot_stellar_gas_hmr_comp(stellar_data, gas_data, snap, weight_norm)

# for snap in eagle_snaps:
#     plot_stellar_hmr("EAGLE", [0, ], snap, weight_norm)
