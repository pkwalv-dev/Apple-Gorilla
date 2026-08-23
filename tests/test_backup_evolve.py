"""Tests for the backup/rollback safety net and the evolve gate."""
from ag import backup
from ag.config import Config
from ag.evolve import _validate_patch, evolve
from ag.model import make_client


def test_snapshot_and_restore(tmp_path, monkeypatch):
    # Snapshot the real evolvable files, mutate one, restore, verify original back.
    cfg = Config()
    target = backup.ROOT / cfg.evolvable_paths[0]
    original = target.read_text()
    snap = backup.snapshot(cfg.evolvable_paths, note="test")
    try:
        target.write_text(original + "\n# mutated by test\n")
        assert target.read_text() != original
        backup.restore(snap)
        assert target.read_text() == original
    finally:
        target.write_text(original)


def test_prune_snapshots_bounds_count(monkeypatch, tmp_path):
    # Redirect the versions dir so we don't touch real state.
    monkeypatch.setattr(backup, "VERSIONS_DIR", tmp_path)
    for i in range(5):
        d = tmp_path / f"snap-{i:03d}"
        d.mkdir()
        (d / "MANIFEST.txt").write_text("note: t\nag/prompts.py\n")
    assert len(backup.list_snapshots()) == 5
    removed = backup.prune_snapshots(keep=2)
    assert removed == 3
    kept = backup.list_snapshots()
    assert kept == ["snap-003", "snap-004"]  # newest two survive


def test_prune_runs_bounds_count(monkeypatch, tmp_path):
    from ag import pipeline
    monkeypatch.setattr(pipeline, "RUNS_DIR", tmp_path)
    for i in range(6):
        (tmp_path / f"2026010{i}-000000-000.json").write_text("{}")
    removed = pipeline._prune_runs(keep=3)
    assert removed == 3
    assert len(list(tmp_path.glob("*.json"))) == 3


def test_git_commit_evolve_safe_outside_repo(monkeypatch, tmp_path):
    # Never raises; returns None when not a git repo (best-effort by design).
    monkeypatch.setattr(backup, "ROOT", tmp_path)
    assert backup.git_commit_evolve("msg", branch="ag/evolve") is None


def test_validate_patch_rejects_bad_python():
    assert _validate_patch("ag/prompts.py", "def (:\n") is not None
    assert _validate_patch("ag/prompts.py", "x = 1\n") is None


def test_validate_patch_rejects_bad_json():
    assert _validate_patch("config.json", "{not json}") is not None
    assert _validate_patch("config.json", '{"ok": true}') is None


def test_evolve_dry_run_no_patches_is_safe():
    # The dry-run evolver proposes nothing; nothing should be adopted or broken.
    cfg = Config()
    client = make_client(dry_run=True)
    result = evolve(client, cfg, dry_run=True)
    assert result.attempted is True
    assert result.adopted is False
    assert result.rolled_back is False


def test_never_disables_evolution():
    cfg = Config()
    cfg.autonomy_level = "never"
    client = make_client(dry_run=True)
    result = evolve(client, cfg, dry_run=True)
    assert result.attempted is False
