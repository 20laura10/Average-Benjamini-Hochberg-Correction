# ======================================
# Settings & Constants
# ======================================
import aaanalysis as aa
import numpy as np
import os
import matplotlib.pyplot as plt

# constant
x = 20

# Path for results
RESULTS_PATH = 'C:/Users/results'

# ======================================
# I Helper Functions
# ======================================
# calculate approximated sbhf
def f_apro(x):
    return np.log(x) + 0.57721


# calculate exact sbhf
def f_exac(x):
    return sum([1 / (k + 1) for k in range(x)])


# ======================================
# II Main Functions
# ======================================
def plot_sbhf(x, filename="sbhf_plot.png"):
    """
    Creates a plot of the approximated and exact sbhf.

    :param x: Constant
    :param filename: File name for saving the plot
    """
    val_apro = [f_apro(i) for i in range(1, x + 1)]
    val_exac = [f_exac(i) for i in range(1, x + 1)]
    val_n = list(range(1, x + 1))

    aa.plot_settings()
    plt.figure(figsize=(10, 6))
    plt.plot(val_n, val_apro, label='Approximated SBHF')
    plt.plot(val_n, val_exac, label='Exact SBHF')
    plt.xlabel('n')
    plt.ylabel('SBHF')
    plt.title('Approximated vs Exact SBHF')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(os.path.join(RESULTS_PATH, filename))
    plt.show()
    plt.close()


# ======================================
# III Test / Caller Functions
# ======================================
def main():
    plot_sbhf(x)


# ======================================
# IV Main
# ======================================
if __name__ == "__main__":
    main()
