import numpy as np
import pytest

from zgq import ZGQConfig, ZGQIndex


@pytest.fixture(scope="session")
def data():
    rng = np.random.RandomState(0)
    vectors = rng.randn(500, 32).astype(np.float32)
    queries = rng.randn(20, 32).astype(np.float32)
    return vectors, queries


@pytest.fixture(scope="session")
def built_index(data):
    vectors, _ = data
    config = ZGQConfig(n_zones=8, M=16, ef_construction=100, ef_search=64, random_state=0)
    return ZGQIndex(config).build(vectors)
