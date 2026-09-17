"""Tests for fleet control, portable bundle, and vetted-GitHub evolution refs."""
import pytest

from ag.config import Config


# --- fleet ------------------------------------------------------------------
def _patch_fleet(monkeypatch, tmp_path):
    from ag import fleet
    monkeypatch.setattr(fleet, "FLEET_DIR", tmp_path)
    monkeypatch.setattr(fleet, "AGENTS_FILE", tmp_path / "agents.jsonl")
    monkeypatch.setattr(fleet, "STOP_FILE", tmp_path / "STOP")


def test_fleet_registry_records_and_lists(monkeypatch, tmp_path):
    from ag import fleet
    _patch_fleet(monkeypatch, tmp_path)
    fleet.record_spawn("root.worker-1", role="worker", parent="root", depth=1)
    fleet.record_spawn("root.researcher-2", role="researcher", parent="root", depth=1)
    agents = fleet.list_agents()
    assert {a.agent for a in agents} == {"root.worker-1", "root.researcher-2"}
    assert fleet.get("root.worker-1").role == "worker"


def test_fleet_disable_and_status(monkeypatch, tmp_path):
    from ag import fleet
    _patch_fleet(monkeypatch, tmp_path)
    fleet.record_spawn("a.b-1", role="w", parent="a", depth=1)
    assert fleet.set_status("a.b-1", "disabled") is True
    assert fleet.is_disabled("a.b-1") is True
    assert fleet.set_status("nope", "disabled") is False


def test_fleet_kill_switch(monkeypatch, tmp_path):
    from ag import fleet
    _patch_fleet(monkeypatch, tmp_path)
    assert fleet.kill_active() is False
    fleet.engage_kill()
    assert fleet.kill_active() is True
    fleet.clear_kill()
    assert fleet.kill_active() is False


def test_kill_switch_blocks_spawn(monkeypatch, tmp_path):
    from ag import fleet, agents
    from ag.permissions import PermissionBroker
    _patch_fleet(monkeypatch, tmp_path)
    fleet.engage_kill()
    broker = PermissionBroker(allow_external_tools=True)
    broker.grant("spawn_agent")
    res = agents.spawn(None, Config(), broker, role="w", task="do x")
    assert "kill switch" in res.output


# --- bundle -----------------------------------------------------------------
def test_bundle_portability_check_passes():
    from ag import bundle
    checks = bundle.check()
    names = {c.name: c for c in checks}
    # The core must be stdlib-only (anthropic allowed) and free of hard-coded paths.
    assert names["stdlib-only core (except anthropic)"].ok, names["stdlib-only core (except anthropic)"].detail
    assert names["no hard-coded absolute paths"].ok, names["no hard-coded absolute paths"].detail
    assert bundle.check_ok(checks) in (True, False)  # returns a bool, no raise


def test_bundle_export_creates_archive(tmp_path):
    import zipfile
    from ag import bundle
    dest = tmp_path / "ag-bundle.zip"
    out = bundle.export(dest)
    assert out.exists()
    with zipfile.ZipFile(out) as z:
        names = z.namelist()
    assert "BUNDLE_MANIFEST.json" in names
    assert any(n.startswith("ag/") and n.endswith(".py") for n in names)
    assert any(n == "config.json" for n in names)


# --- vetted-github evolution refs ------------------------------------------
def test_github_refs_block_disabled_by_default():
    from ag import evolve
    assert evolve._github_refs_block(Config()) == ""


def test_github_refs_block_fetches_when_enabled(monkeypatch):
    from ag import evolve
    from ag.tools import github
    # Stub the fetch so no network is used; assert the block wires repo->briefing.
    monkeypatch.setattr(github, "fetch",
                        lambda repo, path, **k: f"CODE FROM {repo}/{path}")
    cfg = Config(evolve_use_github=True,
                 github_allowlist=["ollama/ollama"],
                 evolve_github_refs=[{"repo": "ollama/ollama", "path": "x.py", "note": "n"}])
    block = evolve._github_refs_block(cfg)
    assert "Vetted reference implementations" in block
    assert "CODE FROM ollama/ollama/x.py" in block
