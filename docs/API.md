# RustPAM API

Minimal API reference for two entry points: Python `OneBatchPAM` and the low-level accelerator `swap_eager`.

## Installation

```bash
pip install rustpam
```

## Quick start

```python
import numpy as np
from rustpam import OneBatchPAM

X = np.random.randn(1000, 10).astype(np.float32)
model = OneBatchPAM(n_medoids=3, distance="euclidean", random_state=0)
model.fit(X)
labels = model.predict(X)
```

## API overview

### OneBatchPAM

One-batch k-medoids (PAM) clustering in Python with a Rust-parallelized core.

Constructor:

```python
OneBatchPAM(
  n_medoids=10,
  distance="euclidean",
  batch_size="auto",
  weighting=True,
  max_iter=100,
  tol=1e-6,
  n_jobs=None,
  random_state=None,
  n_threads=None,
)
```

- Key parameters:
  - `n_medoids`: number of medoids (clusters) to select.
  - `distance`: distance metric passed to `sklearn.metrics.pairwise_distances`.
  - `batch_size`: number of sampled columns; `"auto"` uses a heuristic.
  - `n_threads`: thread count for the Rust kernel (`None` = auto).

Methods:

```python
fit(X) -> self
predict(X) -> np.ndarray  # index of nearest medoid for each sample (0..n_medoids-1)
fit_predict(X) -> np.ndarray  # returns selected medoid indices
```

Attributes after fitting:

- `medoid_indices_`: indices of selected medoids in `X`.
- `labels_`: nearest medoid for each sample (in `[0, n_medoids)`).
- `inertia_`: objective value (sum of distances to nearest medoid).
- `dist_to_nearest_medoid_`: distance to nearest medoid for each sample.
- `n_iter_`: number of swap steps executed internally.
- `cluster_centers_`: medoid feature vectors `X[medoid_indices_]`.
- `solution_`: low-level result dict returned by `swap_eager`.

### swap_eager

Low-level Rust-implemented accelerator. Typically used via `OneBatchPAM`, but can be called directly.

Signature:

```python
swap_eager(
  dist: np.ndarray,            # float32 distance matrix of shape (n, b), C-contiguous
  medoids_init: list[int],     # initial medoid sample indices
  k: int,                      # number of medoids
  max_iter: int,               # maximum swap steps
  n: int,                      # total number of samples
  b: int,                      # number of batch columns (second dim of dist)
  tol: float,                  # relative improvement tolerance
  n_threads: int = 0,          # thread count (0 = auto)
) -> dict
```

Returns a dict with:

- `medoids`: `np.ndarray[int]` selected medoid indices (length `k`).
- `nearest`: `np.ndarray[int]` nearest medoid index for each column/sample in the batch (length `b`).
- `dist_to_nearest`: `np.ndarray[float32]` distance to the nearest (length `b`).
- `loss`: `float` objective value (sum of distances).
- `steps`: `int` number of executed swap steps.

Minimal example:

```python
import numpy as np
from rustpam import swap_eager

n = 100
b = 50
Dist = np.random.rand(n, b).astype(np.float32)
init = np.random.choice(n, 5, replace=False).tolist()
res = swap_eager(Dist, init, k=5, max_iter=100, n=n, b=b, tol=1e-6, n_threads=0)
print(res.keys())  # dict_keys(['medoids', 'nearest', 'dist_to_nearest', 'loss', 'steps'])
```


