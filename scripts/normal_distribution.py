import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests
import aaanalysis as aa

# ======================================
# Helper Functions
# ======================================
def calculate_p_values(mean, std, x):
    return stats.norm.cdf(x, mean, std)


def average_bh_correction(p_values):
    n = len(p_values)
    ranks = np.arange(1, n + 1)
    avg_bh = np.mean(n / ranks)
    return np.clip(p_values * avg_bh, 0, 1)


def plot_density_and_cdf(distributions, x):
    """Plotting of the density and distribution functions."""
    colors_den = ['rosybrown', 'brown', 'darkred', 'salmon']
    colors_cdf = ['slategrey', 'darkblue', 'mediumpurple', 'mediumorchid']

    # plot density functions
    aa.plot_settings()
    plt.figure(figsize=(10, 6))
    for i, dist in enumerate(distributions):
        mean, std = dist['mean'], dist['std']
        pdf = stats.norm.pdf(x, mean, std)
        plt.plot(x, pdf, label=f'Mean={mean}, Std={std}', color=colors_den[i])
    plt.title('Density Functions of Different Normal Distributions')
    plt.xlabel('Range of Values')
    plt.ylabel('Density')
    plt.legend(loc='upper left', frameon=True, fontsize='small')
    plt.grid(True)
    plt.show()

    # plot distribution functions
    aa.plot_settings()
    plt.figure(figsize=(10, 6))
    for i, dist in enumerate(distributions):
        mean, std = dist['mean'], dist['std']
        cdf = stats.norm.cdf(x, mean, std)
        plt.plot(x, cdf, label=f'Mean={mean}, Std={std}', color=colors_cdf[i])
    plt.title('Distribution Functions of Different Normal Distributions')
    plt.xlabel('Range of Values')
    plt.ylabel('Cumulative Probability')
    plt.legend(loc='lower right', frameon=True, fontsize='small')
    plt.grid(True)
    plt.show()


def plot_p_values(original_p, avg_bh_p, bon_p, bh_p, mean, std):
    """
    Visualization of original and corrected p-values.
    :param original_p: Original p-values
    :param avg_bh_p: aBH-corrected p-values
    :param bon_p: Bonferroni-corrected p-values
    :param bh_p: BH-corrected p-values
    :param mean: Mean of the normal distribution
    :param std: Standard deviation of the normal distribution
    """
    aa.plot_settings()
    plt.figure(figsize=(10, 6))
    plt.plot(x, original_p, label='Original P-Values', color='blue')
    plt.plot(x, avg_bh_p, label='aBH Corrected P-Values', color='green')
    plt.plot(x, bon_p, label='Bonferroni Corrected P-Values', color='orange')
    plt.plot(x, bh_p, label='BH Corrected P-Values', color='red')
    plt.axhline(y=0.05, color='purple', linestyle='--', linewidth=1, label='Alpha Value (0.05)')
    plt.title(f'P-Value Correction (Mean={mean}, Std={std})')
    plt.xlabel('Range of Values')
    plt.ylabel('P-Values')
    plt.ylim(0, 1)
    plt.legend(loc='lower right', frameon=True, fontsize='small')
    plt.grid(True)
    plt.show()


# ======================================
# Main Function
# ======================================
if __name__ == "__main__":
    distributions = [
        {'mean': 0, 'std': 1},
        {'mean': 2, 'std': 0.5},
        {'mean': -2, 'std': 1.5},
        {'mean': 4, 'std': 2},
    ]

    x = np.linspace(-10, 10, 1000)

    plot_density_and_cdf(distributions, x)

    for dist in distributions:
        mean, std = dist['mean'], dist['std']

        original_p = calculate_p_values(mean, std, x)
        avg_bh_p = average_bh_correction(original_p)
        bon_p = multipletests(original_p, alpha=0.01, method='bonferroni')[1]
        bh_p = multipletests(original_p, alpha=0.05, method='fdr_bh')[1]

        plot_p_values(original_p, avg_bh_p, bon_p, bh_p, mean, std)