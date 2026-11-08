import zgq


def test_public_api():
    for name in ("ZGQIndex", "ZGQConfig", "ZGQSearch"):
        assert hasattr(zgq, name)
    assert zgq.__version__ == "8.0.0"
