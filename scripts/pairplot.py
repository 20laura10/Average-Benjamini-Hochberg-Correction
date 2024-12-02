# ======================================
# Settings & Constants
# ======================================
import pandas as pd
import os
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests

# Path for results
RESULTS_PATH = 'C:/Users/Laura/Desktop/Praktikum/results'

# ======================================
# I Helper Functions
# ======================================

# Generate p-values
def generate_p_values(size=500, scale=0.2):
    return np.random.exponential(scale=scale, size=size)


# Average Benjamini-Hochberg
def aBH_p_values(p_values):
    n = len(p_values)
    list_p_adj = [p * n for p in p_values]
    avg_bh = sum([n / (i + 1) for i in range(n)]) / n
    list_p_adj_avg_bh = [p * avg_bh for p in p_values]
    return list_p_adj_avg_bh


# Bonferroni
def bonferroni_p_values(p_values):
    bonferroni_corrected = multipletests(p_values, method='bonferroni')[1]
    return bonferroni_corrected


# Benjamini-Hochberg
def bh_p_values(p_values):
    bh_corrected = multipletests(p_values, method='fdr_bh')[1]
    return bh_corrected


# Define plot settings
def plot_settings():
    """Setzt Standard-Einstellungen für Matplotlib-Plots."""
    plt.rcParams.update({
        'figure.figsize': (12, 8),
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
    })


# Calculate correlation between the different methods
def calculate_correlations(df):
    return df.corr()


# ======================================
# II Main Functions
# ======================================
def create_pairplot_with_annotations(df, correlations, title='Pairplot of Original and Corrected P-Values', filename="pairplot"):
    """
    Creates a pairplot with annotations of the correlation values.

    :param df: DataFrame with values to be plotted
    :param correlations: Correlation matrix
    :param title: Title of the plot
    :param filename: File name for saving the plot
    """
    pairplot = sns.pairplot(df, diag_kind='kde', plot_kws={'alpha': 1.0, 'color': 'blue', 'edgecolor': 'none', 's': 40})

    for i, j in zip(*np.triu_indices_from(pairplot.axes, 1)):
        pairplot.axes[i, j].clear()
        pairplot.axes[i, j].annotate(f'{correlations.iloc[i, j]:.2f}', (0.5, 0.5),
                                     textcoords='axes fraction',
                                     ha='center', va='center', fontsize=12, color='black', weight='bold')

    for ax in pairplot.axes.flatten():
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    # plot settings and saving of plot
    plot_settings()
    plt.subplots_adjust(top=0.95, right=0.95, bottom=0.07, left=0.07, hspace=0.15, wspace=0.15)
    plt.suptitle(title, y=1)
    plt.savefig(os.path.join(RESULTS_PATH, filename))
    plt.show()
    plt.close()

# ======================================
# III Test / Caller Functions
# ======================================
def test_create_pairplot():
    """
    testing of the function create_pairplot_with_annotations
    """
    # generate p-values
    p_values = generate_p_values()

    # correction of the generated p-values
    adjusted_bon = bonferroni_p_values(p_values)
    adjusted_bh = bh_p_values(p_values)
    adjusted_abh = aBH_p_values(p_values)

    # prepare data for dataframe
    data = {
        'P-Values': p_values,
        'BON': adjusted_bon,
        'BH': adjusted_bh,
        "aBH": adjusted_abh
    }
    df = pd.DataFrame(data)
    df = df.clip(lower=0, upper=1)

    # calculate correlations for correlation matrix
    correlations = calculate_correlations(df)

    # generate pairplot
    create_pairplot_with_annotations(df, correlations)

# ======================================
# IV Main
# ======================================
if __name__ == "__main__":
    test_create_pairplot()
