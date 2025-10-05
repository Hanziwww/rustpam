"""
Basic test suite for RustPAM - Verify functionality
"""

import numpy as np
import pytest

from rustpam import OneBatchPAM, swap_eager


class TestImports:
    """Test that imports work correctly"""

    def test_import_swap_eager(self):
        """Test that swap_eager can be imported"""
        assert swap_eager is not None

    def test_import_onebatchpam(self):
        """Test that OneBatchPAM can be imported"""
        assert OneBatchPAM is not None


class TestOneBatchPAM:
    """Test OneBatchPAM functionality"""

    @pytest.fixture
    def sample_data(self):
        """Generate test data"""
        np.random.seed(42)
        n_samples = 100
        n_features = 5
        X = np.random.randn(n_samples, n_features).astype(np.float32)
        return X

    def test_model_creation(self):
        """Test that model can be created"""
        model = OneBatchPAM(
            n_medoids=3, distance="euclidean", max_iter=50, random_state=42, n_threads=2
        )
        assert model.n_medoids == 3
        assert model.distance == "euclidean"
        assert model.max_iter == 50
        assert model.random_state == 42

    def test_model_fit(self, sample_data):
        """Test that model can fit data"""
        model = OneBatchPAM(
            n_medoids=3, distance="euclidean", max_iter=50, random_state=42, n_threads=2
        )
        model.fit(sample_data)

        # Check that attributes are set
        assert hasattr(model, "medoid_indices_")
        assert hasattr(model, "labels_")
        assert hasattr(model, "inertia_")
        assert hasattr(model, "n_iter_")

        # Check shapes and types
        assert len(model.medoid_indices_) == 3
        assert len(model.labels_) == len(sample_data)
        assert model.inertia_ >= 0
        assert model.n_iter_ >= 0

    def test_model_predict(self, sample_data):
        """Test that model can predict"""
        model = OneBatchPAM(
            n_medoids=3, distance="euclidean", max_iter=50, random_state=42, n_threads=2
        )
        model.fit(sample_data)

        # Test prediction on new data
        X_test = np.random.randn(20, sample_data.shape[1]).astype(np.float32)
        labels = model.predict(X_test)

        assert len(labels) == len(X_test)
        assert labels.min() >= 0
        assert labels.max() < 3

    def test_reproducibility(self, sample_data):
        """Test that results are reproducible with same random_state"""
        model1 = OneBatchPAM(
            n_medoids=3, distance="euclidean", max_iter=50, random_state=42, n_threads=2
        )
        model1.fit(sample_data)

        model2 = OneBatchPAM(
            n_medoids=3, distance="euclidean", max_iter=50, random_state=42, n_threads=2
        )
        model2.fit(sample_data)

        np.testing.assert_array_equal(model1.medoid_indices_, model2.medoid_indices_)
        np.testing.assert_array_equal(model1.labels_, model2.labels_)
        assert model1.inertia_ == model2.inertia_

    def test_different_k(self, sample_data):
        """Test with different numbers of medoids"""
        for k in [2, 5, 10]:
            model = OneBatchPAM(
                n_medoids=k,
                distance="euclidean",
                max_iter=50,
                random_state=42,
                n_threads=2,
            )
            model.fit(sample_data)
            assert len(model.medoid_indices_) == k
            assert model.labels_.max() < k


class TestSwapEager:
    """Test swap_eager Rust function directly"""

    def test_swap_eager_basic(self):
        """Test basic swap_eager functionality"""
        # Create simple distance matrix
        n = 50
        k = 3
        dist = np.random.rand(n, n).astype(np.float32)
        dist = (dist + dist.T) / 2  # Make symmetric
        np.fill_diagonal(dist, 0)

        medoids_init = np.random.choice(n, k, replace=False).tolist()

        result = swap_eager(
            dist=dist,
            medoids_init=medoids_init,
            k=k,
            max_iter=10,
            n=n,
            b=n,
            tol=0.0,
            n_threads=2,
        )

        assert "medoids" in result
        assert "loss" in result
        assert "steps" in result
        assert len(result["medoids"]) == k
        assert result["loss"] >= 0
