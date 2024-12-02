# ======================================
# Settings & Constants
# ======================================
import numpy as np
import config as cfg

RESULT_PATH = "C:/Users/results"

# ======================================
# I Helper Functions
# ======================================
# array containing the 10 p-values from the paper "Why, when and how to adjust your p-values?"
p_values = np.array([0.0001, 0.001, 0.006, 0.03, 0.095, 0.117, 0.234, 0.552, 0.751, 0.985])


# Average Benjamini-Hochberg correction function
def aBH_p_values(p_values):
    n = len(p_values)
    list_p_adj = [p * n for p in p_values]
    avg_bh = sum([n / (i + 1) for i in range(n)]) / n
    list_p_adj_avg_bh = [p * avg_bh for p in p_values]
    return list_p_adj_avg_bh


# Bonferroni correction function
def bonferroni_p_values(p_values):
    bonferroni_corrected = multipletests(p_values, method='bonferroni')[1]
    return bonferroni_corrected


# Benjamini-Hochberg correction function
def bh_p_values(p_values):
    bh_corrected = multipletests(p_values, method='fdr_bh')[1]
    return bh_corrected

# ======================================
# II Main Functions
# ======================================
def plot_p_value_correction_methods(p_values, filename = "p_values_10_paper"):
    '''
    Function to plot the 10 values from the array p_values.

    :param p_values: Input values
    :param filename: File name for saving the plot
    '''
    adjusted_bon = bonferroni_p_values(p_values)
    adjusted_bh = bh_p_values(p_values)
    adjusted_abh = aBH_p_values(p_values)
    adjusted_abh = np.minimum(adjusted_abh, 1)

    aa.plot_settings()
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    plt.plot(p_values, label='Original P-Values', marker='o')
    plt.plot(adjusted_bon, label='Bonferroni', marker='o')
    plt.plot(adjusted_abh, label='Average Benjamini-Hochberg', marker='o')
    plt.plot(adjusted_bh, label='Benjamini-Hochberg', marker='o')
    plt.xlabel('Index')
    plt.ylabel('P-Values')
    plt.title('Comparison of P-Value Correction Methods')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(os.path.join(RESULT_PATH, filename))
    plt.show()
    plt.close()


# ======================================
# IV Main
# ======================================
# generating the plot by calling the function plot_p_value_correction_methods
cfg.plot_p_value_correction_methods(p_values)
