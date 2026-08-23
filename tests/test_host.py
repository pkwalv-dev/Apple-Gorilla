"""Tests for host introspection and the egress-only web defaults."""
from ag import host
from ag.config import Config


def test_inspect_without_network_probe():
    info = host.inspect(probe_internet=False)
    assert info.cpu_count >= 1
    assert info.system
    assert info.internet is False  # not probed
    d = info.as_dict()
    assert "gpu" in d and "cpu_count" in d


def test_pick_concurrency_is_bounded():
    assert 1 <= host.pick_concurrency(cap=4) <= 4
    assert host.pick_concurrency(cap=1) == 1


def test_has_internet_true(monkeypatch):
    class FakeConn:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(host.socket, "create_connection",
                        lambda *a, **k: FakeConn())
    assert host.has_internet(timeout=0.1) is True


def test_has_internet_false(monkeypatch):
    def boom(*a, **k):
        raise OSError("no net")

    monkeypatch.setattr(host.socket, "create_connection", boom)
    assert host.has_internet(timeout=0.1) is False


def test_web_on_by_default():
    c = Config()
    assert c.allow_web is True        # standing internet access
