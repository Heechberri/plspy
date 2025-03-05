import numpy as np
import scipy.stats

# project imports (if any)
from . import exceptions

# -------------------- RESAMPLING FUNCTIONS --------------------
def resample_without_replacement(matrix, cond_order, C=None, group_num=0, return_indices=False):
    """
    Resamples input matrix without replacement. This implementation uses 
    the condition order to shuffle the rows of the matrix without replacement.
    """
    inds = np.array([i for i in range(len(matrix))])
    grp_split = None
    start = 0
    # Split indices into groups based on cond_order
    for i, group_sizes in enumerate(cond_order):
        group_split = []
        for cond_size in group_sizes:
            group_split.append(inds[start : start + cond_size])
            start += cond_size
        group_split = np.column_stack(group_split)
        if grp_split is None:
            grp_split = group_split
        else:
            grp_split = np.concatenate((grp_split, group_split))
    grp = grp_split
    # Shuffle within each subject's condition
    within_subject_shuffle = np.apply_along_axis(np.random.permutation, axis=1, arr=grp)
    # Shuffle across subjects
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
    """
    inds = np.array([i for i in range(len(matrix))])
    start = 0
    my_resampled = None
    my_shuf_indices = None
    for i, group_sizes in enumerate(cond_order):
        group_split = []
        for cond_size in group_sizes:
            group_split.append(inds[start : start + cond_size])
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

# -------------------- CONFIDENCE INTERVAL METHODS --------------------
class confidence_interval:
    @staticmethod
    def original(matrix, conf=(0.05, 0.95)):
        """
        Computes element-wise confidence intervals using the simple percentile method.
        Assumes that 'matrix' is an array of bootstrap estimates with shape (n_boot, m, n).
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
    def BCa(data, statistic, n_resamples=10000, confidence_level=0.95, random_state=None):
        """
        Computes the bias-corrected and accelerated (BCa) confidence interval using SciPy's bootstrap.
        
        Parameters:
            data : array-like or tuple of array-like
                Data from which to generate bootstrap samples. For multivariate data,
                pass a tuple of arrays.
            statistic : callable
                A function that computes the statistic of interest from the data.
            n_resamples : int, optional
                Number of bootstrap resamples (default 10000).
            confidence_level : float, optional
                The desired confidence level (default 0.95).
            random_state : int or numpy.random.Generator, optional
                For reproducible results.
        
        Returns:
            (lower, upper) : tuple
                The lower and upper bounds of the confidence interval.
        """
        # SciPy's bootstrap requires data in a sequence (e.g., tuple of arrays)
        res = scipy.stats.bootstrap(
            data, statistic, n_resamples=n_resamples, method='BCa',
            confidence_level=confidence_level, random_state=random_state
        )
        return res.confidence_interval.low, res.confidence_interval.high
