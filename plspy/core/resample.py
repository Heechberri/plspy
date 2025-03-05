import numpy as np
import scipy.stats

# project imports
from . import exceptions

def resample_without_replacement(matrix, cond_order, C=None, group_num=0, return_indices=False):
    """
    Resamples input matrix without replacement. This implementation uses
    the condition order to shuffle the rows of the matrix without replacement.
    
    Parameters
    ----------
    matrix : array-like
        Input matrix of arbitrary dimension. Will be cast to a numpy array.
    cond_order : array-like
        Defines the grouping of rows to be shuffled.
    C : array-like, optional
        Condition array (unused in this implementation).
    group_num : int, optional
        Group number; defaults to 0.
    return_indices : bool, optional
        Whether to return the shuffled indices along with the resampled matrix.
    
    Returns
    -------
    resampled : ndarray
        Resampled matrix.
    (Optional) shuf_indices : ndarray
        The order of shuffled indices.
    """
    inds = np.array([i for i in range(len(matrix))])
    grp_split = None
    start = 0
    # Split indices into groups based on cond_order
    for i, group_sizes in enumerate(cond_order):
        group_split = []
        for cond_size in group_sizes:
            group_split.append(inds[start: start + cond_size])
            start += cond_size
        group_split = np.column_stack(group_split)
        if grp_split is None:
            grp_split = group_split
        else:
            grp_split = np.concatenate((grp_split, group_split))
    grp = grp_split
    # Shuffle within each group (row-wise)
    within_subject_shuffle = np.apply_along_axis(np.random.permutation, axis=1, arr=grp)
    # Shuffle across groups
    shuff = np.copy(within_subject_shuffle.T)
    for col in range(grp.shape[1]):
        shuff[col, :] = np.random.permutation(within_subject_shuffle.T[col, :])
    shuf_indices = shuff.ravel()
    resampled = matrix[shuf_indices, :]
    if return_indices:
        return resampled, shuf_indices
    return resampled

def resample_with_replacement(matrix, cond_order, C=None, group_num=0, return_indices=False):
    """
    Resamples input matrix with replacement using the condition order.
    
    Parameters
    ----------
    matrix : array-like
        Input matrix; will be cast to a numpy array.
    cond_order : array-like
        Defines the grouping of rows.
    C : array-like, optional
        Condition array (unused here).
    group_num : int, optional
        Defaults to 0.
    return_indices : bool, optional
        Whether to return the resampled indices.
    
    Returns
    -------
    resampled : ndarray
        Resampled matrix with replacement.
    (Optional) shuf_indices : ndarray
        The resampled order of indices.
    """
    inds = np.array([i for i in range(len(matrix))])
    start = 0
    my_resampled = None
    my_shuf_indices = None
    for i, group_sizes in enumerate(cond_order):
        group_split = []
        for cond_size in group_sizes:
            group_split.append(inds[start: start + cond_size])
            start += cond_size
        group_split = np.column_stack(group_split)
        num_rows = group_split.shape[0]
        shuffled_indices = np.random.choice(num_rows, num_rows, replace=True)
        shuf_cond = []
        for col in range(group_split.shape[1]):
            shuf_cond.append(group_split[shuffled_indices, col])
        shuf_cond = np.vstack(shuf_cond)
        shuf_indices = shuf_cond.ravel()
        resampled = matrix[shuf_indices, :]
        if my_resampled is None:
            my_resampled = resampled
            my_shuf_indices = shuf_indices
        else:
            my_resampled = np.concatenate((my_resampled, resampled))
            my_shuf_indices = np.concatenate((my_shuf_indices, shuf_indices))
    if return_indices:
        return my_resampled, my_shuf_indices
    return my_resampled

# ---------------------------------------------------------------
# Confidence interval functions
# ---------------------------------------------------------------
class confidence_interval:
    @staticmethod
    def original(matrix, conf=(0.05, 0.95)):
        """
        Computes element-wise confidence intervals using the simple percentile method.
        Assumes 'matrix' has shape (n_resamples, m, n), where each (m, n) element 
        contains bootstrap estimates.
        
        Parameters
        ----------
        matrix : ndarray
            Bootstrap distribution.
        conf : tuple, optional
            Nominal quantiles (default (0.05, 0.95)).
        
        Returns
        -------
        lower, upper : tuple of ndarrays
            Lower and upper confidence intervals.
        """
        nrow = matrix.shape[1]
        ncol = matrix.shape[2]
        lower = np.empty((nrow, ncol))
        upper = np.empty((nrow, ncol))
        for i in range(nrow):
            for j in range(ncol):
                lower[i, j] = np.percentile(matrix[:, i, j], conf[0] * 100)
                upper[i, j] = np.percentile(matrix[:, i, j], conf[1] * 100)
        return lower, upper

    @staticmethod
    def BCa(bootstrap_distribution, conf=(0.05, 0.95), orig_estimate_func=np.mean):
        """
        Computes the bias-corrected and accelerated (BCa) confidence interval for each element
        in a precomputed bootstrap distribution. This implementation is based on the formulas in
        Efron and Tibshirani’s "An Introduction to the Bootstrap" (1993).
        
        Parameters
        ----------
        bootstrap_distribution : ndarray
            A 3D array with shape (n_resamples, m, n) containing bootstrap estimates.
        conf : tuple, optional
            Nominal quantiles (default (0.05, 0.95)).
        orig_estimate_func : callable, optional
            Function to compute the original estimate from the bootstrap distribution 
            (default np.mean along axis 0).
        
        Returns
        -------
        lower, upper : tuple of ndarrays
            BCa-adjusted lower and upper confidence intervals, each of shape (m, n).
        """
        n_resamples, m, n = bootstrap_distribution.shape
        lower = np.empty((m, n))
        upper = np.empty((m, n))
        
        # Compute the original estimate for each (i, j)
        orig_estimates = orig_estimate_func(bootstrap_distribution, axis=0)
        
        # Loop over each element
        for i in range(m):
            for j in range(n):
                boot_est = bootstrap_distribution[:, i, j]
                orig_est = orig_estimates[i, j]
                lower[i, j], upper[i, j] = bca_confidence_interval(boot_est, orig_est, conf=conf)
        return lower, upper

def bca_confidence_interval(bootstrap_estimates, orig_estimate, conf=(0.05, 0.95)):
    """
    Compute the BCa confidence interval for a single statistic from a bootstrap distribution.
    
    Parameters
    ----------
    bootstrap_estimates : array-like
        1D array of bootstrap estimates.
    orig_estimate : float
        Statistic computed on the original data.
    conf : tuple, optional
        Nominal lower and upper quantiles (default (0.05, 0.95)).
    
    Returns
    -------
    lower_bound, upper_bound : tuple of floats
        BCa-adjusted lower and upper confidence bounds.
    
    Notes
    -----
    This function implements the bias-corrected and accelerated (BCa) method as described in:
    
        Efron, B. & Tibshirani, R.J. (1993). An Introduction to the Bootstrap. Chapman & Hall/CRC.
    """
    bootstrap_estimates = np.asarray(bootstrap_estimates)
    n = len(bootstrap_estimates)
    
    # Compute bias-correction factor z0
    prop_less = np.sum(bootstrap_estimates < orig_estimate) / n
    z0 = scipy.stats.norm.ppf(prop_less)
    
    # Compute jackknife estimates: leave-one-out averages of bootstrap estimates
    jackknife_estimates = np.array([np.mean(np.delete(bootstrap_estimates, i)) for i in range(n)])
    mean_jack = np.mean(jackknife_estimates)
    
    # Compute acceleration factor a
    num = np.sum((mean_jack - jackknife_estimates) ** 3)
    den = 6.0 * (np.sum((mean_jack - jackknife_estimates) ** 2) ** 1.5)
    a = num / den if den != 0 else 0.0
    
    # Convert nominal quantiles to z-scores
    z_alpha_lower = scipy.stats.norm.ppf(conf[0])
    z_alpha_upper = scipy.stats.norm.ppf(conf[1])
    
    # Adjust quantiles using the BCa formula
    adj_alpha_lower = scipy.stats.norm.cdf(z0 + (z0 + z_alpha_lower) / (1 - a * (z0 + z_alpha_lower)))
    adj_alpha_upper = scipy.stats.norm.cdf(z0 + (z0 + z_alpha_upper) / (1 - a * (z0 + z_alpha_upper)))
    
    # Get corresponding percentiles from bootstrap distribution
    lower_bound = np.percentile(bootstrap_estimates, 100 * adj_alpha_lower)
    upper_bound = np.percentile(bootstrap_estimates, 100 * adj_alpha_upper)
    
    return lower_bound, upper_bound
