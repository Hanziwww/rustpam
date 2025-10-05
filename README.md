# RustPAM

High-performance k-medoids (PAM) clustering with a Rust core, exposed to Python via PyO3. Parallelized with Rayon, offering strong performance and a scikit-learn–style API.

## Quick Start

```python
import numpy as np
from rustpam import OneBatchPAM

# Sample data
X = np.random.randn(1000, 10).astype(np.float32)

# Create and fit model
model = OneBatchPAM(
    n_medoids=3,
    distance="euclidean",
    max_iter=50,
    random_state=42,
    n_threads=2,
)
model.fit(X)

# Results
print("Medoid indices:", model.medoid_indices_)
print("Inertia:", model.inertia_)
print("Iterations:", model.n_iter_)

# Predict new samples
X_new = np.random.randn(20, X.shape[1]).astype(np.float32)
labels = model.predict(X_new)
print("Predicted labels:", labels)
```
