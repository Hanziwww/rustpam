"""One-batch PAM (k-medoids) clustering - Rust Implementation.

This module provides `OneBatchPAM`, a fast, memory-efficient approximation of
PAM (Partitioning Around Medoids). The core algorithm is implemented in Rust
and parallelized with Rayon, providing better performance and parallel scalability
than the original Cython version.
"""

import numpy as np
from sklearn.metrics import pairwise_distances

# Import Rust-implemented core function
# swap_eager is already set up in __init__.py
from rustpam import swap_eager


class OneBatchPAM:
    """One-batch PAM (k-medoids) clustering with Rust parallelization.

    This estimator selects `n_medoids` medoid indices from the input dataset by
    optimizing the total distance to the nearest medoid. To reduce complexity,
    the optimization is performed against a single sampled batch of the distance
    matrix rather than the full pairwise matrix. The core swap routine is
    implemented in Rust + Rayon for high performance.

    Parameters
    ----------
    n_medoids : int, default=10
        Number of medoids (clusters) to select. Must be <= number of samples.

    distance : str or callable, default='euclidean'
        Distance metric passed to `sklearn.metrics.pairwise_distances`.
        Use 'precomputed' to provide your own distance matrix in `fit`.

    batch_size : {'auto'} or int, default='auto'
        Number of candidate columns (points) used when `distance != 'precomputed'`.
        If 'auto', uses a logarithmic heuristic based on dataset size and `n_medoids`.

    weighting : bool, default=True
        If True, reweight columns by current cluster sizes to stabilize optimization.

    max_iter : int, default=100
        Maximum number of swap steps for internal PAM optimization.

    tol : float, default=1e-6
        Relative improvement tolerance to stop the swap phase.

    n_jobs : int or None, default=None
        Number of jobs for `pairwise_distances`. See scikit-learn documentation.

    random_state : int, numpy.random.Generator, or None, default=None
        Random seed or Generator controlling batch sampling and medoid initialization.

    n_threads : int or None, default=None
        Number of threads for Rust-accelerated `swap_eager` kernel. If None, uses default.

    Attributes
    ----------
    medoid_indices_ : ndarray of shape (n_medoids,), dtype=int
        Indices of the selected medoids within the input `X` passed to `fit`.

    labels_ : ndarray of shape (n_samples,), dtype=int
        Index of the nearest medoid (in `[0, n_medoids)`) for each sample.

    inertia_ : float
        Final objective value (sum of distances to assigned medoid).

    dist_to_nearest_medoid_ : ndarray of shape (n_samples,), dtype=float32
        Distance from each sample to its nearest medoid.

    n_iter_ : int
        Number of swap steps performed by the optimizer.

    cluster_centers_ : ndarray of shape (n_medoids, n_features)
        The medoid feature vectors, i.e., `X[medoid_indices_]`.

    solution_ : dict
        Low-level result dictionary returned by `swap_eager`.
        Contains keys: 'medoids', 'nearest', 'loss', 'dist_to_nearest', 'steps'.

    Examples
    --------
    >>> from rustpam import OneBatchPAM
    >>> import numpy as np
    >>> X = np.random.RandomState(0).randn(100, 8).astype(np.float32)
    >>> model = OneBatchPAM(n_medoids=5, random_state=0)
    >>> model.fit(X)
    OneBatchPAM(...)
    >>> model.medoid_indices_.shape
    (5,)
    >>> labels = model.predict(X)
    >>> labels.shape
    (100,)
    """

    def __init__(
        self,
        n_medoids=10,
        distance="euclidean",
        batch_size="auto",
        weighting=True,
        max_iter=100,
        tol=1e-6,
        n_jobs=None,
        random_state=None,
        n_threads=None,
    ):
        """Initialize the estimator.

        Parameters
        ----------
        n_medoids : int, default=10
            Number of medoids (clusters) to select.
        distance : str or callable, default='euclidean'
            Distance metric.
        batch_size : {'auto'} or int, default='auto'
            Candidate batch size.
        weighting : bool, default=True
            Whether to reweight columns by current cluster sizes.
        max_iter : int, default=100
            Maximum number of swap steps.
        tol : float, default=1e-6
            Relative improvement tolerance.
        n_jobs : int or None, default=None
            Parallelism for `pairwise_distances`.
        random_state : int, numpy.random.Generator, or None, default=None
            Random seed.
        n_threads : int or None, default=None
            Number of threads for Rust kernel (None = auto).
        """
        self.n_medoids = n_medoids
        self.distance = distance
        self.batch_size = batch_size
        self.weighting = weighting
        self.max_iter = max_iter
        self.tol = tol
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.n_threads = n_threads

    def fit(self, X):
        """Find the medoids on the provided data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features) or (n_samples, m)
            - If `distance != 'precomputed'`: feature matrix
            - If `distance == 'precomputed'`: precomputed distance matrix

        Returns
        -------
        self : OneBatchPAM
            Fitted estimator.

        Raises
        ------
        ValueError
            If `n_medoids` exceeds the number of samples.
        """
        if self.n_medoids > X.shape[0]:
            raise ValueError("Number of medoids cannot exceed dataset size")

        if X.dtype != np.float32:
            X = X.astype(np.float32)

        rng = np.random.default_rng(self.random_state)

        if self.distance == "precomputed":
            Dist = X
            batch_size = X.shape[1]
        else:
            if self.batch_size == "auto":
                # Use slightly larger batch for stability on small N
                est = int(150.0 * np.log(max(2, X.shape[0] * max(1, self.n_medoids))))
                batch_size = min(max(est, self.n_medoids), X.shape[0])
            else:
                batch_size = int(self.batch_size)
            if batch_size > X.shape[0]:
                batch_size = X.shape[0]
            batch_indexes = rng.choice(X.shape[0], batch_size, replace=False)
            Dist = pairwise_distances(X, X[batch_indexes], metric=self.distance, n_jobs=self.n_jobs)
            maxv = Dist.max()
            if maxv > 0:
                np.divide(Dist, np.float32(maxv), out=Dist, casting="same_kind")

        if self.weighting:
            # Compute argmin using current Dist columns once
            assign = Dist.argmin(1)
            sample_weight = np.zeros(Dist.shape[1], dtype=np.float32)
            unique, counts = np.unique(assign, return_counts=True)
            sample_weight[unique] = counts.astype(np.float32)
            meanw = sample_weight.mean()
            if meanw > 0:
                sample_weight /= meanw
            np.multiply(Dist, sample_weight, out=Dist, casting="same_kind")

        # Ensure C-contiguous float32 distance matrix
        Dist = np.ascontiguousarray(Dist, dtype=np.float32)

        medoids_init = rng.choice(X.shape[0], self.n_medoids, replace=False)
        medoids_init = medoids_init.tolist()

        # Call Rust-implemented swap_eager
        n_threads = 0 if self.n_threads is None else int(self.n_threads)
        self.solution_ = swap_eager(
            Dist,
            medoids_init,
            self.n_medoids,
            self.max_iter,
            X.shape[0],
            batch_size,
            np.float32(self.tol),
            n_threads,
        )

        self.medoid_indices_ = np.array(self.solution_["medoids"])
        self.labels_ = np.array(self.solution_["nearest"])
        self.inertia_ = float(self.solution_["loss"])
        self.dist_to_nearest_medoid_ = np.array(self.solution_["dist_to_nearest"])
        self.n_iter_ = int(self.solution_["steps"])
        self.cluster_centers_ = X[self.medoid_indices_]
        return self

    def predict(self, X):
        """Assign each sample in `X` to the nearest learned medoid.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Feature matrix for which to compute assignments.

        Returns
        -------
        labels : ndarray of shape (n_samples,), dtype=int
            Index of the nearest medoid (in `[0, n_medoids)`) for each sample.
        """
        Dist = pairwise_distances(
            X, self.cluster_centers_, metric=self.distance, n_jobs=self.n_jobs
        )
        return Dist.argmin(1)

    def fit_predict(self, X):
        """Fit the model and return medoid indices.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features) or (n_samples, m)
            Training data or precomputed distances.

        Returns
        -------
        medoid_indices : ndarray of shape (n_medoids,), dtype=int
            Indices of the selected medoids within the input `X`.
        """
        self.fit(X)
        return self.medoid_indices_
