# ======================================
# Imports & Settings
# ======================================
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests
import aaanalysis as aa

# ======================================
# Helper Functions
# ======================================
def plot_histogram(data, title="Normal Distribution"):
    aa.plot_settings()
    plt.figure(figsize=(8, 6))
    plt.hist(data, bins=30, density=True, alpha=0.7, color='b', edgecolor='black')
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.grid(True)
    plt.show()


def plot_adjusted_p_values(p_values, bon_values, avg_bh_values, bh_values, log_scale=False):
    """
    Visualize original and corrected p-values.

    :param p_values: Original p-values
    :param bon_values: Bonferroni- corrected values
    :param avg_bh_values: average Benjamini-Hochberg corrected values
    :param bh_values: BH-corrected values
    :param log_scale: to set y-axis in a logarithmic scale
    """
    aa.plot_settings()
    plt.figure(figsize=(10, 6))
    plt.plot(p_values, label='Original P-Values', marker='o', markersize=2.5, color='blue')
    plt.plot(bon_values, label='Bonferroni', marker='o', markersize=2.5, color='orange')
    plt.plot(avg_bh_values, label='Average Benjamini-Hochberg', marker='o', markersize=2.5, color='green')
    plt.plot(bh_values, label='Benjamini-Hochberg', marker='o', markersize=2.5, color='red')
    plt.axhline(y=0.05, color='purple', linestyle='--', linewidth=1, label='Alpha Value (0.05)')
    plt.xlim(35, 100)
    if log_scale:
        plt.yscale('log')
        plt.ylabel('P-Values [log]')
    else:
        plt.ylabel('P-Values')
    plt.xlabel('Range of P-Values')
    plt.title('Comparison of P-Value Correction Methods - Normal Distribution')
    legend = plt.legend(loc='lower right', frameon=True, fontsize='x-small')
    legend.get_frame().set_edgecolor('black')
    plt.ylim(0, 1)
    plt.grid(True)
    plt.show()


# ======================================
# Main Logic
# ======================================
def main():
    # Parameter
    mu = 0  # Mean
    sigma = 0.5  # Standard deviation
    n = 100  # Number of performed tests

    # generate normal distributed data
    p_values = np.random.normal(mu, sigma, n)

    datensatz_min = np.min(p_values)
    datensatz_max = np.max(p_values)
    datensatz_norm = (p_values - datensatz_min) / (datensatz_max - datensatz_min)
    datensatz_scaled = datensatz_norm * 2 - 1

    plot_histogram(datensatz_scaled, title="Normal Distribution")

    p_values_array = datensatz_scaled[datensatz_scaled <= 1]
    sorted_p_values = np.sort(p_values_array)
    top_100_p_values = sorted_p_values[:100]
    print("Top 100 P-Values:", top_100_p_values)

    # Average Benjamini-Hochberg (aBH)
    n_top = len(top_100_p_values)
    avg_bh = sum([n_top / (i + 1) for i in range(n_top)]) / n_top
    list_p_adj_avg_bh = np.clip([p * avg_bh for p in top_100_p_values], 0, 1)
    print("aBH:", list_p_adj_avg_bh)

    # Bonferroni-Korrektur
    bon_values = multipletests(top_100_p_values, alpha=0.01, method='bonferroni')[1]
    print("Bonferroni-Corrected Values:", bon_values)

    # Benjamini-Hochberg (BH)
    bh_values = multipletests(top_100_p_values, alpha=0.05, method='fdr_bh')[1]
    print("Benjamini-Hochberg (BH) Corrected Values:", bh_values)

    # Diagramm der Korrekturen
    plot_adjusted_p_values(top_100_p_values, bon_values, list_p_adj_avg_bh, bh_values)

    # Log-Skala
    plot_adjusted_p_values(top_100_p_values, bon_values, list_p_adj_avg_bh, bh_values, log_scale=True)


# ======================================
# Execute
# ======================================
if __name__ == "__main__":
    main()
