# ======================================
# Settings & Constants
# ======================================
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests
import os
import aaanalysis as aa

# path for savin the results
RESULTS_PATH = 'C:/Users/Laura/Desktop/Praktikum/results'
LAM = 0.5
SIZE = 1000

# ======================================
# I Helper Functions
# ======================================
def generate_p_values(lam=LAM, size=SIZE):
    p_values = np.random.exponential(scale=1 / lam, size=size)
    p_values_array = p_values[p_values <= 1]
    sorted_p_values = np.sort(p_values_array)
    p_values_100 = sorted_p_values[:100]
    return p_values_100


def aBH_p_values(p_values):
    n = len(p_values)
    avg_bh = sum([n / (i + 1) for i in range(n)]) / n
    list_p_adj_avg_bh = [p * avg_bh for p in p_values]
    return np.clip(list_p_adj_avg_bh, 0, 1)


def bonferroni_p_values(p_values):
    bonferroni_corrected = multipletests(p_values, method='bonferroni')[1]
    return bonferroni_corrected


def bh_p_values(p_values):
    bh_corrected = multipletests(p_values, method='fdr_bh')[1]
    return bh_corrected


def save_and_show_plot(filename, tight_layout=True):
    filepath = os.path.join(RESULTS_PATH, filename)
    if tight_layout:
        plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(filepath)
    plt.show()
    plt.close()


# ======================================
# II Main Functions
# ======================================
def generate_and_plot_exponential_data(lam=0.5, size=1000, log_scale=False, filename="exponential_plot.png"):
    """
    Generates exponential distributed data, corrects them with the BON, BH and aBH correction and plots the results.

    :param lam: Lambda
    :param size: Number of tests performed
    :param filename: Data name
    """
    p_values_100 = generate_p_values(lam=lam, size=size)

    corrected_numbers_100 = aBH_p_values(p_values_100)
    bonferroni_corrected_100 = bonferroni_p_values(p_values_100)
    bh_corrected_100 = bh_p_values(p_values_100)

    aa.plot_settings()
    plt.figure(figsize=(8, 6))
    plt.hist(p_values_100, bins=30, density=True, alpha=0.7, color='b', edgecolor='black')
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title(f'Exponential Distribution (lambda={lam})')
    plt.grid(True)
    save_and_show_plot("hist_" + filename)

    aa.plot_settings()
    plt.figure(figsize=(12, 6))
    plt.plot(p_values_100, label='Original P-Values', marker='o', markersize=2.5, color='blue')
    plt.plot(bonferroni_corrected_100, label='Bonferroni', marker='o', markersize=2.5, color='orange')
    plt.plot(corrected_numbers_100, label='Average Benjamini-Hochberg', marker='o', markersize=2.5, color='green')
    plt.plot(bh_corrected_100, label='Benjamini-Hochberg', marker='o', markersize=2.5, color='red')
    plt.axhline(y=0.05, color='purple', linestyle='--', linewidth=1, label='Alpha Value (0.05)')
    if log_scale:
        plt.yscale('log')
        plt.ylabel('P-Values [log]')
    else:
        plt.ylabel('P-Values')
    plt.xlabel('Range of P-Values')
    plt.ylabel('P-Values')
    plt.title('Comparison of P-Value Correction Methods - Exponential Distribution')
    legend = plt.legend(loc='center right', frameon=True, fontsize='x-small')
    legend.get_frame().set_edgecolor('black')
    plt.ylim(0, 1)
    plt.grid(True)
    save_and_show_plot(filename)

# ======================================
# III Test / Caller Functions
# ======================================
def test_generate_and_plot():
    generate_and_plot_exponential_data(lam=0.5, size=1000, log_scale=False, filename="test_exponential_plot.png")
    generate_and_plot_exponential_data(lam=0.5, size=1000, log_scale=True, filename="test_exponential_plot.png")

# ======================================
# IV Main
# ======================================
if __name__ == "__main__":
    test_generate_and_plot()
