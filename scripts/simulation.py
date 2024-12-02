import numpy as np
from tqdm import tqdm
import scipy.stats as st
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests
import time
from random import random
import aaanalysis as aa

# Settings & Constants
EU_MA_CONSTANT = 0.57721


# I Helper functions
def generate_pvalues_null(num_tests: int = 1000, mu: float = 0, sigma: float = 1) -> np.ndarray:
    """Generate p-values under the null hypothesis. Normally distributed data."""
    data = np.random.normal(loc=mu, scale=sigma, size=num_tests)
    z_scores = (data - mu) / sigma
    p_values = 2 * (1 - st.norm.cdf(np.abs(z_scores)))
    return p_values


def adjust_bh(p_values) -> np.ndarray:
    """Return corrected p-values using Benjamini-Hochberg method."""
    bh_adjusted = multipletests(p_values, method='fdr_bh')[1]
    return bh_adjusted


def adjust_abh(p_values) -> np.ndarray:
    """Return corrected p-values using Average Benjamini-Hochberg method."""
    n = len(p_values)
    avg_bh = sum([n / (i + 1) for i in range(n)]) / n
    abh_adjusted = [p * avg_bh for p in p_values]
    return np.array(abh_adjusted)


def adjust_bonferroni(p_values) -> np.ndarray:
    """Return corrected p-values using Bonferroni correction."""
    bonferroni_adjusted = multipletests(p_values, method='bonferroni')[1]
    return bonferroni_adjusted


# Compute false positives
def compute_false_positive_rates(m: int = 100, k: int = 100, alpha: float = 0.05) -> (np.ndarray, np.ndarray, np.ndarray):
    """Compute false positive rates using different correction methods."""
    bonf = np.zeros(m)
    bh = np.zeros(m)
    avg_bh = np.zeros(m)

    # Fortschrittsanzeige mit tqdm
    for x in tqdm(range(m), desc="False Positive Rates Calculation"):
        num_tests = x + 1
        bonf_thr = alpha / num_tests
        avg_bh_thr = alpha / (np.log(num_tests) + EU_MA_CONSTANT)

        h = 0  # Bonferroni counter
        j = 0  # BH counter
        s = 0  # Average BH counter

        for _ in range(k):
            pvals = generate_pvalues_null(num_tests)
            p_sorted = np.sort(pvals)
            # Bonferroni correction
            h += np.sum(p_sorted <= bonf_thr)
            # Benjamini-Hochberg procedure
            bh_thresholds = (np.arange(1, num_tests + 1) / num_tests) * alpha
            bh_significant = np.sum(p_sorted <= bh_thresholds)
            j += bh_significant
            # Average BH method
            s += np.sum(p_sorted <= avg_bh_thr)

        bonf[x] = (h / (k * num_tests)) * 100.0
        bh[x] = (j / (k * num_tests)) * 100.0
        avg_bh[x] = (s / (k * num_tests)) * 100.0

    return bonf, bh, avg_bh


def plot_false_positives(bonf: np.ndarray, bh: np.ndarray, avg_bh: np.ndarray, m: int) -> None:
    """Plot the percentage of false positives under multiple testing correction methods."""
    aa.plot_settings()
    plt.figure(figsize=(15, 8))
    plt.plot(range(1, m + 1), bonf, label='Bonferroni')
    plt.plot(range(1, m + 1), bh, label='Benjamini-Hochberg')
    plt.plot(range(1, m + 1), avg_bh, label='Average Benjamini-Hochberg')
    plt.xlabel('Number of Tests (k)')
    plt.ylabel('Percentage of False Positives')
    plt.title('Calculation of False Positives of Multiple Testing Corrections')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True, fontsize="x-small")
    plt.subplots_adjust(right=0.75)
    plt.grid(True, which="both", ls="--")
    plt.show()


# Generate mixed p-values for alternative and null hypotheses
def generate_pvalues_mixed(k: int, signum: int) -> (np.ndarray, np.ndarray):
    """
    Generate p-values for tests under both null and alternative hypotheses.
    """
    pvals = np.zeros(k)
    is_significant = np.zeros(k, dtype=bool)
    for jk in range(k):
        if jk < signum:
            # Under alternative hypothesis
            a = -3.7 + random() * 2.57  # Matching the original code's approach
            u = st.norm.cdf(a)
            is_significant[jk] = True
        else:
            # Under null hypothesis
            a = np.random.normal()
            u = st.norm.cdf(a)
        pvals[jk] = u
    return pvals, is_significant


def compute_false_negative_rates(maxk: int, nr: int, alpha: float, signum: int) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """Compute false negative rates using different correction methods."""
    fneg_B = np.zeros(maxk)
    fneg_BH = np.zeros(maxk)
    fneg_aBH = np.zeros(maxk)
    fneg_unc = np.zeros(maxk)

    # Fortschrittsanzeige mit tqdm
    for ik in tqdm(range(maxk), desc="False Negative Rates Calculation"):
        k = ik + 1
        thrB = alpha / k
        thr_aBH = alpha / (np.log(k + 1) + EU_MA_CONSTANT)

        negB = 0.0
        negBH = 0.0
        negaBH = 0.0
        negunc = 0.0

        for _ in range(nr):
            pvals, is_significant = generate_pvalues_mixed(k, signum)
            p_sorted_indices = np.argsort(pvals)
            p_sorted = pvals[p_sorted_indices]
            is_significant_sorted = is_significant[p_sorted_indices]

            # Bonferroni correction
            significant_B = p_sorted <= thrB
            negB += signum - np.sum(significant_B[:signum])

            # Benjamini-Hochberg procedure
            bh_thresholds = (np.arange(1, k + 1) / k) * alpha
            significant_BH = p_sorted <= bh_thresholds
            negBH += signum - np.sum(significant_BH[:signum])

            # Average BH method
            significant_aBH = p_sorted <= thr_aBH
            negaBH += signum - np.sum(significant_aBH[:signum])

            # Uncorrected
            significant_unc = p_sorted <= alpha
            negunc += signum - np.sum(significant_unc[:signum])

        fneg_B[ik] = negB / (nr * signum)
        fneg_BH[ik] = negBH / (nr * signum)
        fneg_aBH[ik] = negaBH / (nr * signum)
        fneg_unc[ik] = negunc / (nr * signum)

    return fneg_B, fneg_BH, fneg_aBH, fneg_unc


def plot_false_negatives(fneg_B: np.ndarray, fneg_BH: np.ndarray, fneg_aBH: np.ndarray, fneg_unc: np.ndarray, maxk: int) -> None:
    """Plot the probability of false negatives under multiple testing correction methods."""
    Ks = range(1, maxk + 1)
    aa.plot_settings()
    plt.figure(figsize=(15, 8))
    plt.title("Calculation of False Negatives of Multiple Testing Corrections")
    plt.plot(Ks, fneg_B, label="Bonferroni")
    plt.plot(Ks, fneg_BH, label="Benjamini-Hochberg")
    plt.plot(Ks, fneg_aBH, label="Average Benjamini-Hochberg")
    plt.plot(Ks, fneg_unc, label="Uncorrected")
    plt.ylabel('Probability of False Negatives')
    plt.xlabel('Number of Tests (k)')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True, fontsize="x-small")
    plt.subplots_adjust(right=0.75)
    plt.grid(True, which="both", ls="--")
    plt.show()


# II Main functions
def simulate_false_positives(m: int = 10, k: int = 100, alpha: float = 0.05) -> None:
    """Simulate and plot false positives."""
    bonf, bh, avg_bh = compute_false_positive_rates(m, k, alpha)
    plot_false_positives(bonf, bh, avg_bh, m)


def simulate_false_negatives(maxk: int = 100, nr: int = 100, alpha: float = 0.05, signum: int = 25) -> None:
    """Simulate and plot false negatives."""
    fneg_B, fneg_BH, fneg_aBH, fneg_unc = compute_false_negative_rates(maxk, nr, alpha, signum)
    plot_false_negatives(fneg_B, fneg_BH, fneg_aBH, fneg_unc, maxk)


# III Test/Caller Functions
def test_simulations():
    """Test simulations with different parameters."""
    #simulate_false_positives(m=1000, k=5000, alpha=0.05)
    simulate_false_negatives(maxk=1000, nr=10, alpha=0.05, signum=25)


# IV Main
def main():
    t0 = time.time()
    test_simulations()
    t1 = time.time()
    print("Total execution time:", t1 - t0, "seconds")


if __name__ == "__main__":
    main()