"""Configuration and filesystem paths for Apple-Gorilla."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List

# Repo root = parent of the `ag/` package directory.
ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.json"
PROFILE_DIR = ROOT / "profile"
STATE_DIR = ROOT / "state"
RUNS_DIR = STATE_DIR / "runs"
VERSIONS_DIR = STATE_DIR / "versions"


@dataclass
class Config:
    """Runtime configuration. Persisted to config.json (itself an evolvable file)."""

    backend: str = "auto"                     # auto | anthropic | ollama | dry
    # When backend == "auto" and no Anthropic creds are present, fall back to this
    # instead of the dry-run stub — so "use Claude when signed in, else run locally"
    # works with one setting. "ollama" | "dry".
    offline_backend: str = "ollama"
    model: str = "claude-opus-5"
    ollama_model: str = "llama3.1"            # used when backend == ollama
    # 127.0.0.1, NOT localhost: on many systems 'localhost' resolves to IPv6 ::1
    # first, but Ollama binds IPv4 only — so 'localhost' wastes ~2s per call failing
    # over ::1 before retrying 127.0.0.1. This hits every model call; keep it numeric.
    ollama_host: str = "http://127.0.0.1:11434"
    ollama_keep_alive: str = "30m"            # keep the model resident between calls
    ollama_options: dict = field(default_factory=dict)  # e.g. {"num_ctx": 8192}
    # When the local Ollama server isn't reachable, try to launch `ollama serve`
    # ourselves (local hosts only) so the offline backend "just works" without the
    # user starting it by hand. Set false to require a manually-started server.
    ollama_autostart: bool = True
    # Local image generation via an Automatic1111/Forge-compatible Stable Diffusion
    # server (POST {sd_host}/sdapi/v1/txt2img). Keyless and offline; you run the SD
    # server separately. Disabled features degrade with a clear "not reachable" note.
    allow_image_gen: bool = True
    sd_host: str = "http://127.0.0.1:7860"
    # When the SD server isn't reachable, try to launch an already-installed one
    # ourselves (local hosts only) with its API enabled, then wait for it to come up.
    # AG does NOT install Stable Diffusion or download models — that is a large,
    # GPU-dependent step you do once yourself. sd_cmd overrides auto-discovery: set it
    # to the launcher (e.g. a path to webui-user.bat / webui.sh, or a full command).
    sd_autostart: bool = True
    sd_cmd: str = ""
    sd_steps: int = 25
    sd_width: int = 512
    sd_height: int = 512
    sd_sampler: str = "Euler a"
    effort: str = "high"                      # low | medium | high | xhigh | max
    # Extended-thinking control (like the Claude app's toggle). "off" disables the
    # model's chain-of-thought (fastest), "on" forces it, "auto" leaves the model to
    # its default. Applies to thinking-capable backends (Qwen3 via Ollama, Claude via
    # the API); ignored by models that don't think.
    think: str = "auto"                       # off | auto | on
    # Interactive pipeline shape. "fast" = ONE model call (skip prompt-engineering
    # AND the self-critique/revise loop) — the responsive default, so AG stays usable
    # and quick to iterate with on a local model. "full" = engineer the prompt, then
    # self-critique and revise (several calls, higher quality, much slower). Web,
    # profile, memory, and conversation context apply in BOTH modes.
    pipeline_mode: str = "fast"               # fast | full
    max_output_tokens: int = 32000            # main answer generation
    meta_output_tokens: int = 16000           # optimizer / critic / evolve calls
    max_iterations: int = 2                   # critique->revise rounds
    critic_pass_threshold: float = 8.0        # 0..10; at/above this we stop iterating
    speed_budget_s: float = 30.0              # target wall-clock for a full speed score
    score_weights: dict = field(default_factory=lambda: {  # directed-evolution axes
        "accuracy": 0.5, "quality": 0.3, "speed": 0.2,
    })
    autonomy_level: str = "guarded"           # manual | guarded | never
    # Split-backend evolution: use a stronger model ONLY to propose self-edits, while
    # the fitness benchmark still runs on the deploy backend — so an adopted change is
    # guaranteed to help the model you actually run offline. Empty = same as `backend`.
    # e.g. backend="ollama", evolver_backend="anthropic": Claude proposes (1 call),
    # Ollama measures (~170 free calls), the gate keeps only what helps Ollama.
    evolver_backend: str = ""                  # "" | anthropic | ollama | dry
    # Empirical self-improvement gate: adopt a self-edit only if it *measurably*
    # scores >= the incumbent on the objective benchmark (ag/bench.py). This is what
    # makes evolution directed rather than blind — never adopt a measured regression.
    fitness_gate: bool = True                  # require a non-regressing benchmark score
    bench_mode: str = "optimize_execute"       # execute | optimize_execute
    bench_max_tasks: int = 0                    # 0 = all tasks; >0 caps for speed
    fitness_tol: float = 0.05                   # absolute noise floor (0..10 scale)
    # A model at temperature > 0 makes fitness a *random variable*, so a single run
    # is a noisy estimate. AG samples the benchmark `bench_samples` times and adopts a
    # change only if its mean beats the incumbent by more than `fitness_k` standard
    # errors of the estimate — statistical significance, not a lucky draw. Set
    # bench_samples=1 to fall back to the cheap single-shot (tolerance-only) gate.
    bench_samples: int = 3                      # benchmark repeats per fitness estimate
    fitness_k: float = 1.0                      # required margin in standard errors
    allow_external_tools: bool = False        # default-deny for Chrome/browser/etc.
    allow_web: bool = True                     # AG's standing internet access
    # The agentic reason->act->observe loop with the SAFE tools on by default: exact
    # arithmetic, memory recall/save, read-only file/dir access, and delegating a
    # focused subtask to a sub-agent. This is what lets AG *act*, not just summarise.
    allow_local_tools: bool = True
    # Arbitrary code execution (python_exec, a real subprocess) stays OPT-IN — it is
    # the one local tool that can change the machine, so it is never granted by default.
    allow_code_exec: bool = False
    max_tool_steps: int = 4                    # reason->act->observe loop bound
    # --- Directed capability acquisition (ag/acquire.py, ag/skills/) ---------
    # When AG lacks a capability a prompt needs, it can AUTHOR a new skill (a tested
    # tool), install its Python deps, or pull vetted code — then use it. Acquired
    # skills persist in a registry and are inherited by sub-agents.
    allow_acquire: bool = True                  # enable self-extension via skills
    acquisition_autonomy: str = "ask"           # ask (confirm install/run) | auto
    skill_test_gate: bool = True                # a new skill must pass its own test
    skill_full_suite_gate: bool = False         # also run AG's full suite (slow, safest)
    max_acquire_per_run: int = 3                # cap skills acquired in one run
    max_subagent_depth: int = 2                 # bound recursive sub-agent spawning
    # Vetted code sources: PyPI is allowed for installs; GitHub fetches are limited to
    # these "owner/repo" prefixes (raw file reads only). You control this list.
    github_allowlist: List[str] = field(default_factory=lambda: [
        "pytorch/pytorch", "huggingface/transformers", "ollama/ollama",
    ])
    use_memory: bool = True                    # recall durable memory into context
    auto_memory: bool = True                    # after a run, distill+store durable facts
    max_history_turns: int = 12                 # conversation turns kept as working memory
    max_memories: int = 200                    # cap on retained memories (legacy/back-compat)
    # --- Layered memory system (ag/memory/) ---------------------------------
    # Recall is by *meaning*: memories carry embeddings and recall blends semantic
    # similarity, keyword overlap, recency, and importance. Model- and engine-agnostic.
    memory_embed_backend: str = "auto"          # auto | ollama | hash | none
    memory_embed_model: str = "nomic-embed-text"  # local Ollama embedding model (keyless)
    memory_weights: dict = field(default_factory=lambda: {  # recall blend
        "semantic": 0.55, "keyword": 0.2, "recency": 0.15, "importance": 0.1,
    })
    memory_recency_halflife_days: float = 30.0  # recency decay half-life
    memory_merge_threshold: float = 0.92        # cosine >= this => near-duplicate, merged
    memory_caps: dict = field(default_factory=lambda: {  # per-layer retention caps
        "episodic": 2000, "semantic": 1000, "procedural": 500,
    })
    memory_reflect: bool = True                 # distill episodes -> facts/procedures
    memory_reflect_every: int = 10              # run reflection every N auto-memory writes
    max_snapshots: int = 20                    # cap on kept source snapshots
    max_runs: int = 100                        # cap on kept run telemetry logs
    evolve_branch: str = "ag/evolve"           # AG's self-commits land here, never main
    evolvable_paths: List[str] = field(default_factory=lambda: [
        "ag/prompts.py",
        "ag/tools/web.py",       # internet code self-improves (highest-churn area)
        "ag/theme.py",           # the web app's look — AG may iterate its own design
        "profile/principles.md",
        "config.json",
    ])

    @classmethod
    def load(cls, path: Path = CONFIG_PATH) -> "Config":
        if path.exists():
            data = json.loads(path.read_text())
            known = {k: data[k] for k in data if k in cls.__dataclass_fields__}
            return cls(**known)
        return cls()

    def save(self, path: Path = CONFIG_PATH) -> None:
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")


def ensure_dirs() -> None:
    for d in (STATE_DIR, RUNS_DIR, VERSIONS_DIR, PROFILE_DIR):
        d.mkdir(parents=True, exist_ok=True)
    for keep in (RUNS_DIR / ".gitkeep", VERSIONS_DIR / ".gitkeep"):
        if not keep.exists():
            keep.write_text("")
