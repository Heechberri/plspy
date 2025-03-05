import abc
import numpy as np
import scipy
import scipy.stats

# project imports
from . import class_functions, exceptions, gsvd, resample

# import scipy.io as sio


class ResampleTest(abc.ABC):
    """Abstract base class for the ResampleTest class set. Forces existence
    of certain functions.
    """

    _subclasses = {}
    pls_alg = None
    # _algs_larger_lvcorr = {"mb", "cmb"}

    # maps abbreviated user-specified classnames to full PLS variant names
    _pls_types = {
        "mct": "Mean-Centering Task PLS",
        # "mct_mg": "Mean-Centering Task PLS - Multi-Group",
        "cst": "Contrast Task PLS",
        "rb": "Regular Behaviour PLS",
        "mb": "Multiblock PLS",
        "csb": "Contrast Behaviour PLS",
        "cmb": "Contrast Multiblock PLS",
    }

    @abc.abstractmethod
    def __str__(self):
        pass

    @abc.abstractmethod
    def __repr__(self):
        pass

    # register valid decorated PLS/resample method as a subclass of ResampleTest
    @classmethod
    def _register_subclass(cls, pls_method):
        def decorator(subclass):
            cls._subclasses[pls_method] = subclass
            return subclass

        return decorator

    # instantiate and return valid registered PLS method specified by user
    @classmethod
    def _create(cls, pls_method, *args, **kwargs):
        if pls_method not in cls._subclasses and pls_method in cls._pls_types:
            raise exceptions.NotImplementedError(
                f"Specified PLS/Resample method {cls._pls_types[pls_method]} "
                "has not yet been implemented."
            )
        elif pls_method not in cls._subclasses:
            raise ValueError(f"Invalid PLS/Resample method {pls_method}")
        cls.pls_alg = pls_method
        return cls._subclasses[pls_method](*args, **kwargs)


@ResampleTest._register_subclass("mct")
@ResampleTest._register_subclass("rb")
@ResampleTest._register_subclass("cst")
@ResampleTest._register_subclass("csb")
@ResampleTest._register_subclass("mb")
@ResampleTest._register_subclass("cmb")
class _ResampleTestTaskPLS(ResampleTest):
    """Class that runs permutation and bootstrap tests for Task PLS. When run,
    this class generates fields for permutation test information (permutation ratio,
    etc.) and for bootstrap test information (confidence intervals, standard errors,
    bootstrap ratios, etc.).
    """

    def __init__(
        self,
        X,
        Y,
        U,
        s,
        V,
        cond_order,
        contrast=None,
        preprocess=None,
        nperm=1000,
        nboot=1000,
        dist=(0.05, 0.95),
        rotate_method=0,
        mctype=0,
    ):
        self.dist = dist

        print(f"PLS ALG: {self.pls_alg}")
        if nperm > 0:
            self.permute_ratio, self.perm_debug_dict = self._permutation_test(
                X,
                Y,
                U,
                s,
                V,
                cond_order,
                nperm,
                self.pls_alg,
                preprocess=preprocess,
                rotate_method=rotate_method,
                mctype=mctype,
                contrast=contrast,
            )

        if nboot > 0:
            if Y is not None:
                (
                    self.conf_ints,
                    self.std_errs,
                    self.boot_ratios,
                    self.LVcorr,
                    self.llcorr,
                    self.ulcorr,
                    self.llcorr_bca,
                    self.ulcorr_bca,
                    self.boot_debug_dict,
                ) = self._bootstrap_test(
                    X,
                    Y,
                    U,
                    s,
                    V,
                    cond_order,
                    nboot,
                    self.pls_alg,
                    preprocess=preprocess,
                    rotate_method=rotate_method,
                    dist=self.dist,
                    contrast=contrast,
                )
            else:
                (
                    self.conf_ints,
                    self.std_errs,
                    self.boot_ratios,
                    self.boot_debug_dict,
                ) = self._bootstrap_test(
                    X,
                    Y,
                    U,
                    s,
                    V,
                    cond_order,
                    nboot,
                    self.pls_alg,
                    preprocess=preprocess,
                    rotate_method=rotate_method,
                    mctype=mctype,
                    dist=self.dist,
                    contrast=contrast,
                )

    @staticmethod
    def _permutation_test(
        X,
        Y,
        U,
        s,
        V,
        cond_order,
        niter,
        pls_alg,
        preprocess=None,
        contrast=None,
        rotate_method=0,
        mctype=0,
        threshold=1e-12,
    ):
        """Run permutation test on X. Resamples X (without replacement) based
        on condition order, runs PLS on resampled matrix, and computes the
        element-wise permutation ratio.
        """
        greatersum = np.zeros(s.shape)
        s[np.abs(s) < threshold] = 0
        debug = True
        debug_dict = {}
        indices = np.empty((niter, X.shape[0]))
        if debug:
            sum_perm = np.empty(niter)
            sum_s = np.empty(niter)
            s_list = np.empty((niter, s.shape[0]))

        print("----Running Permutation Test----\n")
        for i in range(niter):
            if (i + 1) % 50 == 0:
                print(f"Iteration {i + 1}")
            X_new, inds = resample.resample_without_replacement(
                X, cond_order, return_indices=True
            )
            indices[i] = inds

            if Y is not None:
                Y_new = Y  # or resample.resample_without_replacement(Y, cond_order)

            if Y is None:
                permuted = preprocess(
                    X_new, cond_order, mctype=mctype, return_means=False
                )
            else:
                permuted = preprocess(X_new, Y_new, cond_order)

            if debug:
                sum_perm[i] = np.sum(np.power(permuted, 2))

            if rotate_method == 0:
                if contrast is None:
                    VS_hat = permuted.T @ U
                    s_hat = np.sqrt(np.sum(VS_hat ** 2, axis=0))
                else:
                    inpt = contrast.T @ permuted
                    s_hat = np.linalg.svd(inpt, compute_uv=False)
            elif rotate_method == 1:
                if contrast is not None:
                    U_hat, s_hat, V_hat = class_functions._run_pls_contrast(
                        permuted, contrast
                    )
                else:
                    U_hat, s_hat, V_hat = np.linalg.svd(
                        permuted, full_matrices=False
                    )
                    V_hat = V_hat.T
                U_bar, s_bar, V_bar = np.linalg.svd(
                    U.T @ U_hat, full_matrices=False
                )
                V_bar = V_bar.T
                rot = V_bar @ U_bar.T
                U_rot = (U_hat * s_hat) @ rot
                s_rot = np.sqrt(np.sum(np.power(U_rot, 2), axis=0))
                s_hat = np.copy(s_rot)
            elif rotate_method == 2:
                US_hat = permuted.T @ U
                s_hat = np.sqrt(np.sum(np.power(US_hat, 2), axis=0))
                V_hat_der = US_hat / s_hat
                U_hat = (np.linalg.inv(np.diag(s_hat)) @ (V_hat_der.T @ permuted.T)).T
                V_hat = V_hat_der
            else:
                raise exceptions.NotImplementedError(
                    f"Specified rotation method ({rotate_method}) has not been implemented."
                )
            s_hat[np.abs(s_hat) < threshold] = 0
            greatersum += s_hat >= s
            if debug:
                s_list[i:,] = s_hat
                sum_s[i] = np.sum(np.power(s_hat, 2))

        permute_ratio = greatersum / niter

        print(f"real s: {s}")
        print(f"ratio: {permute_ratio}")
        if debug:
            debug_dict["s_list"] = s_list
            debug_dict["sum_s"] = sum_perm
            debug_dict["sum_perm"] = sum_s
            debug_dict["indices"] = indices
            return (permute_ratio, debug_dict)
        return permute_ratio

    @staticmethod
    def _bootstrap_test(
        X,
        Y,
        U,
        s,
        V,
        cond_order,
        niter,
        pls_alg,
        preprocess=None,
        rotate_method=0,
        mctype=0,
        dist=(0.05, 0.95),
        contrast=None,
    ):
        """Runs a bootstrap estimation on X matrix. Resamples X with
        replacement according to the condition order, runs PLS on the
        resampled X matrices, and computes confidence intervals, standard
        errors, and bootstrap ratios.
        """
        debug = True
        debug_dict = {}

        left_sv_sampled = np.empty((niter, U.shape[0], U.shape[1]))
        right_sv_sampled = np.empty((niter, V.shape[0], V.shape[1]))
        indices = np.empty((niter, X.shape[0]))

        if Y is not None:
            ncols = np.product(cond_order.shape) * Y.shape[1]
            if pls_alg in ["mb", "cmb"]:
                ncols = X.shape[1]
            LVcorr = np.empty(
                (
                    niter,
                    np.product(cond_order.shape) * Y.shape[1],
                    ncols,
                )
            )

        print("----Running Bootstrap Test----\n")
        for i in range(niter):
            if (i + 1) % 50 == 0:
                print(f"Iteration {i + 1}")

            X_new, inds = resample.resample_with_replacement(
                X, cond_order, return_indices=True
            )
            indices[i] = inds

            if Y is not None:
                Y_new = Y[inds, :]

            if Y is None:
                permuted = preprocess(
                    X_new, cond_order, mctype=mctype, return_means=False
                )
            else:
                permuted = preprocess(X_new, Y_new, cond_order)

            if rotate_method == 0:
                U_hat = (np.dot(V.T, permuted.T)).T
                VS_hat = permuted.T @ U
                base = np.sqrt(np.sum(VS_hat ** 2, axis=0))
                base[base == 0] = 1
                V_hat = VS_hat / base
                V_hat[:, base == 1] = 0
            elif rotate_method == 1:
                U_hat, s_hat, V_hat = np.linalg.svd(
                    permuted, full_matrices=False
                )
                V_hat = V_hat.T
                U_bar, s_bar, V_bar = np.linalg.svd(
                    U.T @ U_hat, full_matrices=False
                )
                V_bar = V_bar.T
                rot = V_bar @ U_bar.T
                U_rot = (U_hat * s_hat) @ rot
                permuted_rot = U_rot @ permuted
                s_rot = np.sqrt(np.sum(np.power(permuted_rot.T, 2), axis=0))
                s_hat = np.copy(s_rot)
            elif rotate_method == 2:
                VS_hat = permuted.T @ U
                s_hat = np.sqrt(np.sum(np.power(VS_hat, 2), axis=0))
                V_hat_der = VS_hat / s_hat
                U_hat = (np.linalg.inv(np.diag(s_hat)) @ (V_hat_der.T @ permuted.T)).T
                V_hat = V_hat_der
            else:
                raise exceptions.NotImplementedError(
                    f"Specified rotation method ({rotate_method}) has not been implemented."
                )

            left_sv_sampled[i] = U_hat
            right_sv_sampled[i] = VS_hat
            if Y is not None:
                X_hat_latent = class_functions._compute_X_latents(X_new, V_hat)
                LVcorr[i] = class_functions._compute_corr(X_hat_latent, Y_new, cond_order)
                left_sv_sampled[i] = LVcorr[i]

        conf_int = resample.confidence_interval(left_sv_sampled, conf=dist)
        std_errs = np.std(right_sv_sampled, axis=0)
        boot_ratios = np.divide(V * s, std_errs)

        if debug:
            debug_dict["left_sv_sampled"] = left_sv_sampled
            debug_dict["right_sv_sampled"] = right_sv_sampled
            debug_dict["indices"] = indices

        if Y is None:
            return (conf_int, std_errs, boot_ratios, debug_dict)
        else:
            llcorr, ulcorr = resample.confidence_interval(LVcorr, conf=dist)
            llcorr_bca, ulcorr_bca = resample.confidence_interval.BCa(LVcorr, conf=dist)
            return (
                conf_int,
                std_errs,
                boot_ratios,
                LVcorr,
                llcorr,
                ulcorr,
                llcorr_bca,
                ulcorr_bca,
                debug_dict,
            )

    def __repr__(self):
        stg = ""
        stg += "Permutation Test Results\n"
        stg += "------------------------\n\n"
        stg += f"Ratio: {self.permute_ratio}\n\n"
        stg += "Bootstrap Test Results\n"
        stg += "----------------------\n\n"
        stg += f"Element-wise Confidence Interval: {self.dist}\n"
        stg += "\nLower CI: \n"
        stg += str(self.conf_ints[0])
        stg += "\n\nUpper CI: \n"
        stg += str(self.conf_ints[1])
        stg += "\n\nStandard Errors:\n"
        stg += str(self.std_errs)
        stg += "\n\nBootstrap Ratios:\n"
        stg += str(self.boot_ratios)
        return stg

    def __str__(self):
        stg = ""
        stg += "Permutation Test Results\n"
        stg += "------------------------\n\n"
        stg += f"Ratio: {self.permute_ratio}\n\n"
        stg += "Bootstrap Test Results\n"
        stg += "----------------------\n\n"
        stg += f"Element-wise Confidence Interval: {self.dist}\n"
        stg += "\nLower CI: \n"
        stg += str(self.conf_ints[0])
        stg += "\n\nUpper CI: \n"
        stg += str(self.conf_ints[1])
        stg += "\n\nStandard Errors:\n"
        stg += str(self.std_errs)
        stg += "\n\nBootstrap Ratios:\n"
        stg += str(self.boot_ratios)
        return stg
