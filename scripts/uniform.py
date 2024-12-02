# ======================================
# Settings & Imports
# ======================================
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests
import aaanalysis as aa

# settings for reproducibility
np.random.seed(42)

# ======================================
# I Helper Functions
# ======================================
def plot_uniform_histogram(data):
    """Generates histogram of the uniform distributed data."""
    aa.plot_settings()
    plt.figure(figsize=(8, 6))
    plt.hist(data, bins=30, density=True, alpha=0.7, color='b', edgecolor='black')
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title('Uniform Distribution')
    plt.grid(True)
    plt.show()


def calculate_adjustments(p_values):
    """
    Calculates p-values of the different correction methods.

    :param p_values: Array of p-values
    """
    n = len(p_values)

    # Average Benjamini-Hochberg
    avg_bh = sum([n / (i + 1) for i in range(n)]) / n
    list_p_adj_avg_bh = np.clip([p * avg_bh for p in p_values], 0, 1)  # Werte auf [0, 1] begrenzen

    # Bonferroni correction
    bonferroni_corrected = multipletests(p_values, alpha=0.01, method='bonferroni')[1]

    # Benjamini-Hochberg correction
    bh_corrected = multipletests(p_values, alpha=0.05, method='fdr_bh')[1]

    return {
        'aBH': list_p_adj_avg_bh,
        'Bonferroni': bonferroni_corrected,
        'BH': bh_corrected
    }


def plot_adjusted_p_values(p_values, adjustments, log_scale=False):
    """
    Visualise p-values before and after correction.

    :param p_values: Original p-values
    :param adjustments: Dictionary containing the corrected p-values
    """
    aa.plot_settings()
    plt.figure(figsize=(10, 6))
    plt.plot(p_values, label='Original P-Values', marker='o', markersize=2.5, color='blue')
    plt.plot(adjustments['Bonferroni'], label='Bonferroni', marker='o', markersize=2.5, color='orange')
    plt.plot(adjustments['aBH'], label='Average Benjamini-Hochberg', marker='o', markersize=2.5, color='green')
    plt.plot(adjustments['BH'], label='Benjamini-Hochberg', marker='o', markersize=2.5, color='red')
    plt.axhline(y=0.05, color='purple', linestyle='--', linewidth=1, label='Alpha Value (0.05)')
    plt.xlabel('Range of P-Values')
    plt.ylabel('P-Values')
    if log_scale:
        plt.yscale('log')
        plt.ylabel('P-Values [log]')
    else:
        plt.ylabel('P-Values')
    plt.title('Comparison of P-Value Correction Methods - Uniform Distribution')
    legend = plt.legend(loc='lower right', frameon=True, fontsize='x-small')
    legend.get_frame().set_edgecolor('black')
    plt.ylim(0, 1)
    plt.grid(True)
    plt.show()

# ======================================
# II Main Function
# ======================================
def main():
    # generate data
    size = 100
    min_val = 0
    max_val = 1
    data = np.random.uniform(min_val, max_val, size)

    plot_uniform_histogram(data)
    sorted_data = np.sort(data)
    top_10_values = sorted_data[:100]

    adjustments = calculate_adjustments(top_10_values)
    print("aBH:", adjustments['aBH'])
    print("Bonferroni:", adjustments['Bonferroni'])
    print("BH:", adjustments['BH'])

    plot_adjusted_p_values(top_10_values, adjustments, log_scale=False)
    plot_adjusted_p_values(top_10_values, adjustments, log_scale=True)

# ======================================
# IV Main
# ======================================
if __name__ == "__main__":
    main()
