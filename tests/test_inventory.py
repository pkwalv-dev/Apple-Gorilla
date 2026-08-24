"""Tests for the tool/app inventory + integration/friction ratings."""
from ag import inventory
from ag.config import Config
from ag.permissions import GATED


def test_inventory_covers_core_tools():
    reports = inventory.inventory(Config())
    names = " ".join(r.name for r in reports)
    for expected in ("anthropic", "ollama", "dry-run", "web search", "sub-agents",
                     "host introspection", "pytest"):
        assert expected in names, f"inventory missing {expected!r}"


def test_every_report_is_well_formed():
    for r in inventory.inventory(Config()):
        assert r.status in ("available", "degraded", "unavailable")
        assert 0 <= r.integration <= 10
        assert 0 <= r.friction <= 10
        assert r.category in ("backend", "retrieval", "agent", "host",
                              "capability", "gate")
        assert r.notes


def test_scaffolded_capabilities_are_reported_unavailable():
    reports = {r.name: r for r in inventory.inventory(Config())}
    # Gated caps that still have no driver surface as scaffolded/unavailable. The
    # now-wired ones (network, filesystem_read, code_exec) are reported elsewhere.
    for cap in GATED:
        if cap in inventory._WIRED_CAPS:
            continue
        assert cap in reports, f"scaffolded capability {cap!r} not inventoried"
        assert reports[cap].status == "unavailable"


def test_new_capabilities_are_inventoried():
    names = " ".join(r.name for r in inventory.inventory(Config()))
    assert "local tools" in names and "persistent memory" in names


def test_web_status_tracks_allow_web():
    on = {r.name: r for r in inventory.inventory(_cfg(allow_web=True))}
    off = {r.name: r for r in inventory.inventory(_cfg(allow_web=False))}
    assert on["web search + fetch"].status == "available"
    assert off["web search + fetch"].status == "degraded"


def test_summary_aggregates_and_picks_worst():
    data = inventory.summary(Config())
    assert data["counts"]["total"] == len(data["tools"])
    assert 0 <= data["avg_friction"] <= 10
    assert 0 <= data["avg_integration"] <= 10
    # highest_friction_wired must be a tool that is NOT unavailable.
    if data["highest_friction_wired"]:
        wired = {t["name"]: t for t in data["tools"] if t["status"] != "unavailable"}
        assert data["highest_friction_wired"] in wired


def _cfg(**kw):
    c = Config()
    for k, v in kw.items():
        setattr(c, k, v)
    return c
