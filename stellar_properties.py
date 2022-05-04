import numpy as np
import matplotlib.pyplot as plt
from flare import plt as flareplt
from utils import mkdir, plot_meidan_stat, age2z


def birth_met(stellar_data, snap, weight_norm):

    # Define redshift
    z = float(snap.split("z")[-1].replace("p", "."))
    
    # Extract arrays 
    ages = stellar_data["Particle,S_Age"]
    zs = np.zeros(ages.size)
    mets = stellar_data["Particle,S_Z_smooth"]
    w = np.zeros(ages.size)

    # Extract weights for each particle
    for igal in range(stellar_data["begin"].size):

        # Extract galaxy range
        b = stellar_data["begin"][igal]
        e = b + stellar_data["stride"][igal]

        # Set weights for these particles
        w[b: e] = stellar_data["weights"][igal]

    # Convert ages to birth redshifts
    for ind, a in enumerate(ages):

        # Compute this redshift
        zs[ind] = age2z(a, z)

    # Set up the plot
    fig = plt.figure(figsize=(3.5, 3.5))
    ax = fig.add_subplot(111)

    im = ax.hexbin(zs, mets, gridsize=50,
                   mincnt=np.min(w) - (0.1 * np.min(w)),
                   C=w,
                   extent=[4.5, 15, -5, 0],
                   reduce_C_function=np.sum, yscale='log',
                   norm=weight_norm, linewidths=0.2,
                   cmap='viridis')

    ax.set_ylabel(r"$Z_{\mathrm{birth}}$")
    ax.set_xlabel(r"$z_{\mathrm{birth}}$")

    # Save figure
    mkdir("plots/stellar_evo/")
    fig.savefig("plots/stellar_evo/stellar_birthZ_%s.png" % snap,
                bbox_inches="tight")

