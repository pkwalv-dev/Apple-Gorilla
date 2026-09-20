"""OS adaptation — work on the machine you are actually on, including unknown ones.

`host.py` *describes* the machine (CPU, RAM, GPU, connectivity). This module lets AG
*adapt* to it: which shell to invoke, which package manager installs software, how
paths and line endings behave, whether the filesystem is case-sensitive, whether we
are inside WSL or a container or Termux, and where the user's directories are.

The design principle is the same one that makes the format layer work: **never
branch on a hard-coded list of operating systems and fail on the ones that aren't in
it.** Every fact is a probe with a confidence, and an unrecognised platform gets a
documented POSIX-assumption fallback plus an honest statement of what AG doesn't
know about it. That is what "access any OS that can exist" means in practice — not a
claim to have tested every OS, but a layer that degrades into probing and asking
rather than crashing or silently doing the wrong thing.

Nothing here executes anything. It reads `platform`, `os`, `sys`, `shutil.which`, and
a few well-known files. Acting on what it finds still goes through the permission
broker like everything else.
"""
from __future__ import annotations

import os
import platform
import shutil
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Package managers, in the order we prefer them per platform family. Each entry is
# (binary, install_template, search_template, kind). Data, not code, so a new
# distro/manager is one line rather than a new branch.
PACKAGE_MANAGERS: Tuple[Tuple[str, str, str, str], ...] = (
    ("apt-get", "apt-get install -y {pkg}", "apt-cache search {q}", "system"),
    ("apt", "apt install -y {pkg}", "apt search {q}", "system"),
    ("dnf", "dnf install -y {pkg}", "dnf search {q}", "system"),
    ("yum", "yum install -y {pkg}", "yum search {q}", "system"),
    ("zypper", "zypper --non-interactive install {pkg}", "zypper search {q}", "system"),
    ("pacman", "pacman -S --noconfirm {pkg}", "pacman -Ss {q}", "system"),
    ("apk", "apk add {pkg}", "apk search {q}", "system"),
    ("emerge", "emerge {pkg}", "emerge --search {q}", "system"),
    ("xbps-install", "xbps-install -y {pkg}", "xbps-query -Rs {q}", "system"),
    ("nix-env", "nix-env -iA nixpkgs.{pkg}", "nix search nixpkgs {q}", "system"),
    ("brew", "brew install {pkg}", "brew search {q}", "system"),
    ("port", "port install {pkg}", "port search {q}", "system"),
    ("pkg", "pkg install -y {pkg}", "pkg search {q}", "system"),       # FreeBSD/Termux
    ("pkg_add", "pkg_add {pkg}", "pkg_info -Q {q}", "system"),          # OpenBSD
    ("winget", "winget install --silent {pkg}", "winget search {q}", "system"),
    ("choco", "choco install -y {pkg}", "choco search {q}", "system"),
    ("scoop", "scoop install {pkg}", "scoop search {q}", "system"),
    ("snap", "snap install {pkg}", "snap find {q}", "sandboxed"),
    ("flatpak", "flatpak install -y {pkg}", "flatpak search {q}", "sandboxed"),
)

# Commands that open a file/URL in the platform's default handler.
OPENERS: Dict[str, str] = {
    "Darwin": "open", "Windows": "start", "Linux": "xdg-open",
    "FreeBSD": "xdg-open", "OpenBSD": "xdg-open", "NetBSD": "xdg-open",
    "SunOS": "xdg-open", "AIX": "xdg-open",
}


@dataclass
class Capability:
    """One adaptable fact about this machine, with how we learned it."""

    name: str
    value: object
    confidence: float = 1.0
    evidence: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Platform:
    """Everything AG needs to behave correctly on this machine."""

    family: str = "unknown"          # linux | macos | windows | bsd | unknown
    system: str = ""                 # platform.system() raw
    release: str = ""
    machine: str = ""                # x86_64, arm64, ...
    python: str = ""
    # Environment flavour — these change what "the machine" even means.
    container: str = ""              # docker | podman | kubernetes | lxc | ""
    wsl: str = ""                    # wsl1 | wsl2 | ""
    termux: bool = False             # Android via Termux
    ci: str = ""                     # github | gitlab | generic | ""
    is_admin: bool = False
    # Execution surface.
    shell: str = ""
    shell_flavor: str = ""           # posix | powershell | cmd
    package_managers: List[dict] = field(default_factory=list)
    open_cmd: str = ""
    python_exe: str = ""
    # Filesystem behaviour.
    path_sep: str = "/"
    line_ending: str = "\n"
    case_sensitive_fs: bool = True
    max_path: int = 4096
    home: str = ""
    temp: str = ""
    config_dir: str = ""
    data_dir: str = ""
    # Honesty channel.
    confidence: float = 1.0
    unknowns: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)

    # --- the adaptation API -------------------------------------------------
    def shell_command(self, command: str) -> List[str]:
        """Wrap a command string in the right shell invocation for this platform."""
        if self.shell_flavor == "powershell":
            return [self.shell, "-NoProfile", "-NonInteractive", "-Command", command]
        if self.shell_flavor == "cmd":
            return [self.shell, "/c", command]
        return [self.shell or "/bin/sh", "-c", command]

    def install_command(self, package: str) -> Optional[str]:
        """The command that would install `package` here, or None if unknown.

        Returns a *string to show the user or run behind a grant* — this module
        never installs anything itself.
        """
        if not self.package_managers:
            return None
        pm = self.package_managers[0]
        return pm["install"].format(pkg=package)

    def python_install_command(self, package: str) -> str:
        """Installing a Python dep is the one case that is uniform everywhere."""
        return f"{self.python_exe or sys.executable} -m pip install {package}"

    def normalize_path(self, p: str) -> str:
        """Best-effort path normalisation across conventions, including WSL.

        Under WSL a user will paste `C:\\Users\\x`; that path exists, but as
        `/mnt/c/Users/x`. Translating it is the difference between "file not found"
        and working.
        """
        p = os.path.expanduser(os.path.expandvars(p.strip().strip('"\'')))
        if self.wsl and len(p) > 2 and p[1] == ":" and p[0].isalpha():
            drive = p[0].lower()
            return "/mnt/" + drive + p[2:].replace("\\", "/")
        if self.family == "windows":
            return p.replace("/", "\\")
        return p

    def summary(self) -> str:
        bits = [self.family or "unknown", self.machine or "?"]
        if self.wsl:
            bits.append(self.wsl)
        if self.container:
            bits.append(self.container)
        if self.termux:
            bits.append("termux/android")
        if self.ci:
            bits.append(f"ci:{self.ci}")
        return " · ".join(bits)


# --------------------------------------------------------------------------- #
# Probes
# --------------------------------------------------------------------------- #

def _family(system: str) -> Tuple[str, float, List[str]]:
    s = (system or "").lower()
    if s == "linux":
        return "linux", 1.0, []
    if s == "darwin":
        return "macos", 1.0, []
    if s == "windows":
        return "windows", 1.0, []
    if s in ("freebsd", "openbsd", "netbsd", "dragonfly"):
        return "bsd", 1.0, []
    if s in ("sunos", "aix", "haiku", "cygwin", "os400", "emscripten"):
        # Real, rare, POSIX-ish. Say what we assume, and what we can't promise.
        return "unknown", 0.6, [
            f"{system} is POSIX-like but not one AG has specific handling for; "
            "treating it as POSIX (forward slashes, LF, /bin/sh)"]
    return "unknown", 0.3, [
        f"platform.system() returned {system!r}, which AG does not recognise. "
        "Assuming POSIX conventions. Ask the user before doing anything "
        "platform-specific here."]


def _detect_wsl() -> Tuple[str, str]:
    """WSL1 vs WSL2 — they differ in filesystem performance and GPU access."""
    if platform.system() != "Linux":
        return "", ""
    env = os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP")
    rel = ""
    try:
        rel = Path("/proc/sys/kernel/osrelease").read_text(errors="replace").lower()
    except OSError:
        pass
    if "microsoft" in rel or env:
        # WSL2 runs a real Linux kernel (version 4.19+ with "WSL2" or a -microsoft-
        # standard suffix); WSL1 reports a translated "Microsoft" release.
        if "wsl2" in rel or "microsoft-standard" in rel:
            return "wsl2", f"osrelease={rel.strip()[:60]}"
        return ("wsl2" if env and "wsl1" not in rel else "wsl1",
                f"osrelease={rel.strip()[:60]} env={bool(env)}")
    return "", ""


def _detect_container() -> Tuple[str, str]:
    if Path("/.dockerenv").exists():
        return "docker", "/.dockerenv present"
    if os.environ.get("KUBERNETES_SERVICE_HOST"):
        return "kubernetes", "KUBERNETES_SERVICE_HOST set"
    if os.environ.get("container"):
        return os.environ["container"], "container env var"
    try:
        cg = Path("/proc/1/cgroup").read_text(errors="replace")
        for marker, name in (("docker", "docker"), ("kubepods", "kubernetes"),
                             ("lxc", "lxc"), ("libpod", "podman")):
            if marker in cg:
                return name, f"/proc/1/cgroup mentions {marker}"
    except OSError:
        pass
    return "", ""


def _detect_termux() -> bool:
    return ("com.termux" in os.environ.get("PREFIX", "")
            or Path("/data/data/com.termux").exists()
            or "ANDROID_ROOT" in os.environ)


def _detect_ci() -> str:
    if os.environ.get("GITHUB_ACTIONS"):
        return "github"
    if os.environ.get("GITLAB_CI"):
        return "gitlab"
    if os.environ.get("CI"):
        return "generic"
    return ""


def _detect_shell(family: str) -> Tuple[str, str, str]:
    """Return (shell_path, flavor, evidence)."""
    if family == "windows":
        pwsh = shutil.which("pwsh") or shutil.which("powershell")
        if pwsh:
            return pwsh, "powershell", "found pwsh/powershell on PATH"
        comspec = os.environ.get("COMSPEC") or "cmd.exe"
        return comspec, "cmd", "COMSPEC"
    sh = os.environ.get("SHELL")
    if sh and Path(sh).exists():
        return sh, "posix", "$SHELL"
    for cand in ("/bin/bash", "/bin/sh", "/system/bin/sh",
                 "/data/data/com.termux/files/usr/bin/bash"):
        if Path(cand).exists():
            return cand, "posix", f"{cand} exists"
    return "/bin/sh", "posix", "assumed (nothing else found)"


def _detect_package_managers() -> List[dict]:
    out = []
    for binary, install, search, kind in PACKAGE_MANAGERS:
        path = shutil.which(binary)
        if path:
            out.append({"name": binary, "path": path, "install": install,
                        "search": search, "kind": kind})
    return out


def _case_sensitive_fs(tmp: Path) -> Tuple[bool, str]:
    """Probe rather than assume: macOS is usually case-INsensitive but can be either,
    and a Linux box can have a case-insensitive mount. Guessing corrupts file lookups.
    """
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(prefix="AGcase", dir=str(tmp),
                                         delete=False) as f:
            p = Path(f.name)
        try:
            swapped = Path(str(p).replace("AGcase", "agCASE"))
            result = not swapped.exists()
            return result, "probed with a temp file"
        finally:
            p.unlink(missing_ok=True)
    except Exception:
        return platform.system() not in ("Windows", "Darwin"), "assumed from OS"


def _is_admin() -> bool:
    try:
        if platform.system() == "Windows":
            import ctypes
            return bool(ctypes.windll.shell32.IsUserAnAdmin())
        return os.geteuid() == 0
    except Exception:
        return False


def _dirs(family: str) -> Tuple[str, str, str, str]:
    """(home, temp, config_dir, data_dir) following each platform's conventions."""
    import tempfile
    home = str(Path.home()) if _safe_home() else ""
    tmp = tempfile.gettempdir()
    if family == "windows":
        cfg = os.environ.get("APPDATA") or str(Path(home) / "AppData/Roaming")
        data = os.environ.get("LOCALAPPDATA") or str(Path(home) / "AppData/Local")
    elif family == "macos":
        cfg = str(Path(home) / "Library/Application Support")
        data = cfg
    else:
        cfg = os.environ.get("XDG_CONFIG_HOME") or str(Path(home) / ".config")
        data = os.environ.get("XDG_DATA_HOME") or str(Path(home) / ".local/share")
    return home, tmp, cfg, data


def _safe_home() -> bool:
    try:
        Path.home()
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #

_CACHE: Optional[Platform] = None


def detect(*, refresh: bool = False) -> Platform:
    """Probe this machine. Cached — nothing here changes during a process's life."""
    global _CACHE
    if _CACHE is not None and not refresh:
        return _CACHE

    system = platform.system()
    family, conf, unknowns = _family(system)
    wsl, wsl_ev = _detect_wsl()
    container, container_ev = _detect_container()
    termux = _detect_termux()
    shell, flavor, shell_ev = _detect_shell(family)
    home, tmp, cfg_dir, data_dir = _dirs(family)
    case_sensitive, case_ev = _case_sensitive_fs(Path(tmp))
    notes: List[str] = []

    if wsl:
        notes.append(
            f"Running under {wsl.upper()} ({wsl_ev}). Windows drives are mounted "
            "under /mnt/<drive>; a Windows-style path the user gives will be "
            "translated automatically. Windows executables (.exe) are callable "
            "directly via interop.")
    if container:
        notes.append(
            f"Running inside a {container} container ({container_ev}). Changes "
            "outside mounted volumes are ephemeral, and installed packages will "
            "not survive a container restart — say so before installing anything.")
    if termux:
        notes.append(
            "Running under Termux on Android. There is no systemd, no root by "
            "default, GPU compute is unavailable, and package names differ from "
            "desktop Linux. Prefer pure-Python solutions here.")
    if family == "unknown":
        notes.append(
            "This platform is not one AG has specific handling for. It will use "
            "POSIX defaults and should ask for guidance before running any "
            "platform-specific command.")

    p = Platform(
        family=family, system=system, release=platform.release(),
        machine=platform.machine(), python=platform.python_version(),
        container=container, wsl=wsl, termux=termux, ci=_detect_ci(),
        is_admin=_is_admin(),
        shell=shell, shell_flavor=flavor,
        package_managers=_detect_package_managers(),
        open_cmd=OPENERS.get(system, "xdg-open"),
        python_exe=sys.executable,
        path_sep=os.sep, line_ending="\r\n" if family == "windows" else "\n",
        case_sensitive_fs=case_sensitive,
        max_path=260 if family == "windows" else 4096,
        home=home, temp=tmp, config_dir=cfg_dir, data_dir=data_dir,
        confidence=conf, unknowns=unknowns, notes=notes,
    )
    if not p.package_managers:
        p.unknowns.append(
            "No package manager found on PATH — AG cannot install system software "
            "here and should ask the user to install dependencies manually.")
    _CACHE = p
    return p


def capabilities() -> List[Capability]:
    """The probe results as individual facts with their evidence."""
    p = detect()
    out = [
        Capability("family", p.family, p.confidence, f"platform.system()={p.system!r}"),
        Capability("shell", p.shell, 1.0, f"flavor={p.shell_flavor}"),
        Capability("package_manager",
                   p.package_managers[0]["name"] if p.package_managers else None,
                   1.0 if p.package_managers else 0.0,
                   f"{len(p.package_managers)} found on PATH"),
        Capability("case_sensitive_fs", p.case_sensitive_fs, 0.95, "probed"),
        Capability("admin", p.is_admin, 1.0, "euid/IsUserAnAdmin"),
        Capability("container", p.container or None, 1.0, "cgroup/env probe"),
        Capability("wsl", p.wsl or None, 1.0, "osrelease probe"),
        Capability("gpu_available", _gpu_hint(), 0.7, "vendor tool on PATH"),
    ]
    return out


def _gpu_hint() -> str:
    """Which accelerator toolchain is present — decides whether LoRA can run at all."""
    if shutil.which("nvidia-smi"):
        return "nvidia"
    if shutil.which("rocm-smi"):
        return "amd-rocm"
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "apple-metal"
    return "none-detected"


def report() -> str:
    """Human-readable adaptation report for `ag osinfo` and the doctor."""
    p = detect()
    lines = [
        f"platform:         {p.summary()}",
        f"system:           {p.system} {p.release} ({p.machine})",
        f"python:           {p.python} at {p.python_exe}",
        f"shell:            {p.shell} [{p.shell_flavor}]",
        f"package managers: " + (", ".join(m["name"] for m in p.package_managers)
                                 or "none found"),
        f"open command:     {p.open_cmd}",
        f"filesystem:       {'case-sensitive' if p.case_sensitive_fs else 'case-INsensitive'}"
        f", sep {p.path_sep!r}, newline {p.line_ending!r}",
        f"privileges:       {'administrator/root' if p.is_admin else 'normal user'}",
        f"accelerator:      {_gpu_hint()}",
        f"home / temp:      {p.home}  |  {p.temp}",
        f"config / data:    {p.config_dir}  |  {p.data_dir}",
        f"detection conf.:  {p.confidence:.2f}",
    ]
    for n in p.notes:
        lines.append(f"  note: {n}")
    for u in p.unknowns:
        lines.append(f"  UNKNOWN: {u}")
    return "\n".join(lines)


def guidance_text() -> str:
    """A compact briefing injected into the model's system prompt.

    The model otherwise assumes it is on generic Linux and confidently suggests
    `apt install` on a Mac, or backslash paths on WSL. This is cheap to include and
    removes a whole class of wrong answers.
    """
    p = detect()
    pm = p.package_managers[0]["name"] if p.package_managers else "none"
    lines = [
        "# This machine (adapt your commands to it)",
        f"- OS: {p.summary()}; shell: {p.shell} ({p.shell_flavor}); "
        f"package manager: {pm}",
        f"- Paths use {p.path_sep!r}; text files end lines with "
        f"{'CRLF' if p.line_ending == chr(13) + chr(10) else 'LF'}; the filesystem is "
        f"{'case-sensitive' if p.case_sensitive_fs else 'case-insensitive'}.",
        f"- Python is {p.python} at {p.python_exe}; install Python packages with "
        f"`{p.python_exe} -m pip install <pkg>`.",
    ]
    if not p.is_admin:
        lines.append("- You do NOT have root/administrator rights; prefer user-level "
                     "installs and say when something needs elevation.")
    for n in p.notes:
        lines.append(f"- {n}")
    for u in p.unknowns:
        lines.append(f"- UNCERTAIN: {u}")
    return "\n".join(lines)
