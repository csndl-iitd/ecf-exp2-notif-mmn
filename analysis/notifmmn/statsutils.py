import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests

def corrected_p(pvals, method='fdr_bh'):
    """Apply multiple comparisons correction to p-values.

    Args:
        pvals (array-like): Array of p-values to correct.
        method (str): Correction method ('fdr_bh', 'bonferroni', etc.).
    Returns:
        np.ndarray: Array of corrected p-values.
    """
    pvals = np.asarray(pvals)
    _, pvals_corrected, _, _ = multipletests(pvals, method=method)
    return pvals_corrected
