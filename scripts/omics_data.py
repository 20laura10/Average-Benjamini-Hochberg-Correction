# ======================================
# Imports & Settings
# ======================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests
import os
import aaanalysis as aa

# File path and sheet name
RESULTS_PATH = 'C:/Users/Laura/Desktop/Praktikum/results' # Path in which the figures are saved

file_path = "C:/Users/Laura/Documents/Uni/Master/Forschungspraktikum Bioinformatik/omics_data/test.xlsx"
file = "test" # name of the file
distribution_filename = 'test_distribution'
sheet_name = "Sheet1"

# ======================================
# Helper Functions
# ======================================
def read_data(file_path, sheet_name):
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        df['-log10 p-value A/B'] = pd.to_numeric(df['-log10 p-value A/B'], errors='coerce')
        df['pValue'] = 10 ** (-df['-log10 p-value A/B'])
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error reading the Excel file: {e}")
        return None


def process_p_values(df):
    log_p_values = df['-log10 p-value A/B']
    p_values_array = log_p_values.to_numpy()
    p_values = p_values_array[p_values_array <= 1]
    sorted_p_values = np.sort(p_values)
    top_100_p_values = sorted_p_values[:100]
    return top_100_p_values


def apply_aBH_method(top_100_p_values):
    n = len(top_100_p_values)
    avg_bh = np.mean([n / (i + 1) for i in range(n)])
    list_p_adj_avg_bh = [p * avg_bh for p in top_100_p_values]
    corrected_numbers = np.array([min(1, num) for num in list_p_adj_avg_bh])
    return corrected_numbers


def apply_correction_methods(top_100_p_values):
    bonferroni_corrected = multipletests(top_100_p_values, method='bonferroni')[1]
    bh_corrected = multipletests(top_100_p_values, method='fdr_bh')[1]
    return bonferroni_corrected, bh_corrected


def calculate_percentages_for_correction_methods(top_100_p_values):
    original_percentage = np.sum(top_100_p_values < 0.05) / len(top_100_p_values) * 100
    aBH_corrected = apply_aBH_method(top_100_p_values)
    aBH_corrected_percentage = np.sum(aBH_corrected < 0.05) / len(aBH_corrected) * 100
    _, bh_corrected = apply_correction_methods(top_100_p_values)
    bh_corrected_percentage = np.sum(bh_corrected < 0.05) / len(bh_corrected) * 100
    return {
        "original_percentage": original_percentage,
        "aBH_corrected_percentage": aBH_corrected_percentage,
        "bh_corrected_percentage": bh_corrected_percentage
    }

# ======================================
# II Main Functions
# ======================================
def plot_p_values(top_100_p_values, bonferroni_corrected, corrected_numbers, bh_corrected, filename="filename"):
    aa.plot_settings()
    plt.figure(figsize=(10, 6))
    plt.plot(top_100_p_values, label='Original p-values', marker='o', markersize=2.5, color="blue")
    plt.plot(bonferroni_corrected, label='Bonferroni', marker='o', markersize=2.5, color="orange")
    plt.plot(corrected_numbers, label='Average Benjamini-Hochberg', marker='o', markersize=2.5, color="green")
    plt.plot(bh_corrected, label='Benjamini-Hochberg', marker='o', markersize=2.5, color="red")
    plt.axhline(y=0.05, color='purple', linestyle='--', linewidth=1, label='Alpha Value (0.05)')
    plt.xlabel('Range of P-Values')
    plt.ylabel('P-Values')
    plt.title(f'Comparison of P-Value Correction Methods, Dataset "{filename}"', pad=20)
    plt.legend(loc='upper right', frameon=True, fontsize='small')
    plt.ylim(0, 1)
    plt.grid(True)

    save_path = os.path.join(RESULTS_PATH, f"{filename}.png")
    plt.savefig(save_path)
    plt.show()
    plt.close()
    print(f"Plot saved to {save_path}")


def plot_histogram(df, column_name, dataset_name, distributionname):
    aa.plot_settings()
    plt.figure(figsize=(10, 6))
    plt.hist(df[column_name], bins=30, edgecolor='black', alpha=0.7, density=True)
    plt.title(f'Distribution of {column_name} - {dataset_name}')
    plt.xlabel(column_name)
    plt.ylabel('Frequency')
    plt.grid(True)
    save_path = os.path.join('C:/Users/Laura/Desktop/Praktikum/results', f"{distributionname}.png")
    plt.savefig(save_path)
    plt.show()
    plt.close()
    print(f"Histogram saved to {save_path}")

# ======================================
# IV Main
# ======================================
def main():
    # Read and process data
    df = read_data(file_path, sheet_name)
    if df is None:
        return

    top_100_p_values = process_p_values(df)

    # Apply correction methods
    corrected_numbers = apply_aBH_method(top_100_p_values)
    bonferroni_corrected, bh_corrected = apply_correction_methods(top_100_p_values)

    # Calculate percentages
    percentages = calculate_percentages_for_correction_methods(top_100_p_values)

    print("Percentage of original p-values < 0.05:", percentages["original_percentage"])
    print("Percentage of aBH corrected p-values < 0.05:", percentages["aBH_corrected_percentage"])
    print("Percentage of BH corrected p-values < 0.05:", percentages["bh_corrected_percentage"])

    # Plot and save the results
    plot_p_values(top_100_p_values, bonferroni_corrected, corrected_numbers, bh_corrected, filename=file)

    # Distribution
    plot_histogram(df, column_name='pValue', dataset_name=file, distributionname=distribution_filename)


if __name__ == "__main__":
    main()
