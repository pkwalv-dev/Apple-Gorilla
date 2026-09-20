"""Platform adaptation. The point is to *probe* rather than assume, and to be
explicit about what is unknown instead of guessing confidently."""
from __future__ import annotations

from ag import osadapt


def test_detect_returns_a_populated_platform():
    p = osadapt.detect()
    assert p.system and p.machine and p.python_exe
    assert p.family in ("linux", "macos", "windows", "bsd", "android", "unknown")
    assert p.shell_flavor in ("posix", "powershell", "cmd")
    assert isinstance(p.case_sensitive_fs, bool)


def test_detect_is_cached_but_refreshable():
    a = osadapt.detect()
    assert osadapt.detect() is a, "repeated detection must not re-probe"
    assert osadapt.detect(refresh=True) is not a


def test_case_sensitivity_is_probed_not_inferred_from_os_name():
    """Filesystem case behaviour is a property of the MOUNT, not the OS: macOS can
    be case-sensitive and Linux can host a case-insensitive volume. It is cheap to
    test with a real file, so guessing from the platform name is indefensible."""
    p = osadapt.detect(refresh=True)
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        probe = os.path.join(d, "AgCaseProbe.tmp")
        with open(probe, "w") as fh:
            fh.write("x")
        actual = not os.path.exists(os.path.join(d, "agcaseprobe.tmp"))
    assert p.case_sensitive_fs == actual


def test_shell_command_wraps_for_the_local_shell():
    cmd = osadapt.detect().shell_command("echo hi")
    assert isinstance(cmd, list) and cmd
    assert any("echo hi" in part for part in cmd)


def test_install_command_names_a_real_manager():
    p = osadapt.detect()
    cmd = p.install_command("ripgrep")          # a ready-to-run string
    assert "ripgrep" in cmd
    if p.package_managers:
        # The manager is a record (name/path/install template), so the command is
        # generated from a verified-present binary rather than a hard-coded guess.
        assert p.package_managers[0]["name"] in cmd


def test_normalize_path_is_identity_for_native_paths():
    p = osadapt.detect()
    if p.shell_flavor == "posix":
        for native in ("/tmp/x", "relative/path.txt", "./a.txt"):
            assert p.normalize_path(native) == native


def test_windows_paths_are_translated_under_wsl():
    """A user on WSL pastes C:\\Users\\... constantly; failing on it is a bad answer
    to a question AG can simply understand."""
    import dataclasses
    p = dataclasses.replace(osadapt.detect(), family="linux", wsl="WSL2",
                            path_sep="/", shell_flavor="posix")
    assert p.normalize_path(r"C:\Users\bob\notes.txt") == "/mnt/c/Users/bob/notes.txt"
    assert p.normalize_path(r"D:\data") == "/mnt/d/data"
    assert p.normalize_path("/home/bob") == "/home/bob", "native paths untouched"


def test_capabilities_carry_evidence_and_confidence():
    """Every claim about the machine is auditable: what was concluded, how sure,
    and on what evidence. An unexplained assertion cannot be checked by a human."""
    caps = osadapt.capabilities()
    assert caps
    for c in caps:
        assert c.name and c.evidence, f"{c.name} states no evidence"
        assert 0.0 <= c.confidence <= 1.0


def test_report_and_guidance_are_human_readable():
    rep = osadapt.report()
    assert "platform" in rep.lower()
    g = osadapt.guidance_text()
    assert len(g) < 1200, "a prompt injection must stay small; it is paid per token"
    assert "machine" in g.lower()


def test_unknown_platform_degrades_to_posix_with_declared_unknowns(monkeypatch):
    """An OS AG has never heard of must not crash it. POSIX is the right default,
    and the uncertainty has to be *stated* rather than hidden."""
    import platform as _plat
    monkeypatch.setattr(_plat, "system", lambda: "Plan9")
    monkeypatch.setattr(_plat, "release", lambda: "4")
    p = osadapt.detect(refresh=True)
    try:
        # Honest about the identity...
        assert p.family == "unknown"
        assert p.confidence < 0.5, "an unknown OS must lower the confidence it reports"
        assert any("plan9" in u.lower() for u in p.unknowns), \
            "an unrecognised OS must be named in `unknowns`, not silently absorbed"
        # ...but still USABLE: POSIX conventions are the right bet, and a degraded
        # answer beats a crash. Adaptability means never having no answer at all.
        assert p.shell_flavor == "posix"
        assert p.path_sep == "/"
        assert p.shell_command("echo hi")
    finally:
        monkeypatch.undo()
        osadapt.detect(refresh=True)  # restore real detection for later tests


def test_as_dict_is_json_serialisable():
    import json
    json.dumps(osadapt.detect().as_dict())
