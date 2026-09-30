import warnings

import pytest

from tcga_brca.artifact import save_bundle
from tcga_brca.demo import demo_bundle, synthetic_cohort


@pytest.fixture(scope="session")
def bundle():
    """A model bundle trained with the production pipeline on synthetic data."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)  # sklearn 'penalty' deprecation
        return demo_bundle()


@pytest.fixture(scope="session")
def model_path(bundle, tmp_path_factory):
    return save_bundle(bundle, tmp_path_factory.mktemp("models") / "demo.joblib")


@pytest.fixture
def batch():
    """60 new samples from the same distribution as the training data."""
    X, _ = synthetic_cohort(n_samples=60, seed=1)
    return X


@pytest.fixture
def shifted_batch():
    """60 new samples with a batch effect on half of the genes."""
    X, _ = synthetic_cohort(n_samples=60, seed=2, shift=1.5)
    return X
