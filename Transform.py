import numpy as np
from scipy.spatial import distance_matrix
from skimage.transform import warp

__all__ = ["TPSTransform", "CSRBFTransform"]

class BaseTransform():
    """
    Base transform for thin-plate spline transform and compactly support radial basis function
    for image registration.
    """
    def __init__(self):
        pass

    def transform(self, image, src, dst, **kwargs):
        """
        Given two matching sets of points, source and destination, this function
        transform the given image using TPS transform.

        Parameters
        ----------
        image : array_like
        src: source point set, array_like
        dst: destination point set, array_line

        Returns
        -------
        transformed image: array_like
        """
        src = np.asarray(src)
        dst = np.asarray(dst)

        if src.shape != dst.shape:
            raise ValueError(f"Shape of `src` and `dst` didn't match, {src.shape} != {dst.shape}")
        if src.shape[0] < 3 or dst.shape[0] < 3:
            raise ValueError("Need at least 3 points in in `src` and `dst`")

        n, d = dst.shape

        dist = distance_matrix(dst, dst) # skimage.transform.warp need inverse mapping
        K = self.kernel(dist)
        P = np.hstack([np.ones((n, 1)), dst])

        n_plus_d = n + d + 1
        L = np.zeros((n_plus_d, n_plus_d), dtype=np.float64)
        L[:n, :n] = K
        L[:n, n:] = P
        L[n:, :n] = P.T

        Y = np.vstack([src, np.zeros((d + 1, d))])

        try:
            W = np.linalg.solve(L, Y)
        except np.linalg.LinAlgError:
            raise RuntimeError("Matrix L is singular and cannot be solved")

        def mapping(coords):
            coords = np.array(coords)
            _n, _d = coords.shape
            if _d != d or coords.ndim != 2:
                raise ValueError(f"Input `coords` must have shape (N, {d})")
            dist = distance_matrix(coords, dst)
            K = self.kernel(dist)
            P = np.hstack([np.ones((_n, 1)), coords])
            return np.matmul(np.hstack([K, P]), W)

        warped = warp(image, mapping, **kwargs).astype(image.dtype)
        return warped

    def kernel(self, r):
        pass

class TPSTransform(BaseTransform):
    """
    Thin-plate spline (TPS) transform.

    Given two matching sets of points, source and destination, this class
    transform the given image using TPS transform.

    Modified from skimage.transform.ThinPlateSplineTransform class.
    https://github.com/scikit-image/scikit-image/blob/v0.25.2/skimage/transform/_thin_plate_splines.py

    Bookstein F L. Principal warps: Thin-plate splines and the decomposition of deformations.
    DOI: 10.1109/34.24792
    """
    def __init__(self):
        super().__init__()

    @staticmethod
    def kernel(r):
        r_sq = r**2
        return r_sq * np.log(r_sq + 1e-8)

class CSRBFTransform(BaseTransform):
    """
    Compactly support radial basis function (CSRBF) transform.

    Given two matching sets of points, source and destination, this class
    transform the given image using CSRBF transform.

    Fornefett M, Rohr K, Stiehl H S. Radial basis functions with compact support for
    elastic registration of medical images.
    DOI: https://doi.org/10.1016/S0262-8856(00)00057-3
    """
    def __init__(self, R = 50):
        super().__init__()
        self.R = R # the radius of locality

    def setRadius(self, R):
        self.R = R

    def kernel(self, r):
        r = np.asarray(r)
        _r = r/self.R
        mask = _r < 1.0
        phi = np.zeros_like(_r)
        phi[mask] = (1 - _r[mask])**4 * (4 * _r[mask] + 1)
        return phi