from zgq import ZGQConfig, ZGQIndex


def test_config_defaults():
    c = ZGQConfig()
    assert c.n_zones == "auto"
    assert c.use_hierarchy is True
    assert c.M == 16
    assert c.ef_construction == 200
    assert c.ef_search == 64
    assert c.metric == "l2"
    assert c.random_state == 42


def test_index_uses_default_config():
    index = ZGQIndex()
    assert isinstance(index.config, ZGQConfig)
    assert index.is_built is False
