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

    # Default to the LOCAL model so AG never spends Claude API tokens unless the user
    # explicitly asks for it (pick "Claude cloud" in the model menu, or set backend
    # "anthropic"/"auto"). "auto" = use Claude when signed in, else offline_backend.
    backend: str = "ollama"                   # ollama | anthropic | auto | dry
    # When backend == "auto" and no Anthropic creds are present, fall back to this
    # instead of the dry-run stub. "ollama" | "dry".
    offline_backend: str = "ollama"
    model: str = "claude-opus-4-8"            # Claude model, used only on anthropic/auto
    # The PRIMARY local brain: a strong instruct model that reasons, uses tools, and
    # answers every turn. Dual-model routing (see ag/routing.py) lets it call the
    # specialist below, and the runtime falls back to it when the primary hard-fails.
    ollama_model: str = "qwen3:8b"            # used when backend == ollama
    specialist_model: str = "ag-coder-abliterated:latest"  # abliterated fallback/specialist
    # 127.0.0.1, NOT localhost: on many systems 'localhost' resolves to IPv6 ::1
    # first, but Ollama binds IPv4 only — so 'localhost' wastes ~2s per call failing
    # over ::1 before retrying 127.0.0.1. This hits every model call; keep it numeric.
    ollama_host: str = "http://127.0.0.1:11434"
    # Dual-model routing: the primary reasons and answers; the specialist is consulted
    # when the capability doc favours it for the task, and the runtime falls back to it
    # when the primary errors, empties, refuses, or returns a non-answer.
    model_routing: bool = True                # consult/fall back to the specialist model
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
    # --- ComfyUI backend: unfiltered, open-weight image AND video generation -----
    # A1111 can only run SD-family checkpoints. The current open-weight models that
    # carry no safety filter in the weights — Chroma1-HD for images (8.9B, Apache-2.0,
    # a de-distilled FLUX.1-schnell retrained with no safety filter, and the only one
    # of these that restores real CFG and negative prompts) and Wan 2.2 TI2V-5B for
    # video (Apache-2.0, text- and image-to-video in one checkpoint, the largest video
    # model that fits an 8GB card) — both run under ComfyUI, so AG speaks its API too.
    #
    # "auto" prefers ComfyUI when it answers and falls back to A1111, so an existing
    # SD install keeps working untouched.
    image_backend: str = "auto"               # auto | comfy | a1111
    allow_video_gen: bool = True
    comfy_host: str = "http://127.0.0.1:8188"
    comfy_autostart: bool = True
    comfy_cmd: str = ""                       # overrides auto-discovery of main.py
    # Workflows are DATA, not code: AG substitutes the prompt/seed/size into a graph
    # exported from ComfyUI itself ("Export (API)"). Drop a replacement of the same
    # name in state/workflows/ and it wins over the shipped one — so a new model is a
    # new JSON file, not a patch to AG.
    comfy_image_workflow: str = "chroma1hd_txt2img"
    comfy_video_workflow: str = "wan22_ti2v_txt2vid"
    video_width: int = 704
    video_height: int = 400
    video_frames: int = 49                    # ~2s at 24fps; 121 is Wan 2.2's 5s
    video_fps: int = 24
    video_steps: int = 20
    effort: str = "high"                      # low | medium | high | xhigh | max
    # Extended-thinking control (like the Claude app's toggle). "off" disables the
    # model's chain-of-thought (fastest), "on" forces it, "auto" leaves the model to
    # its default. Applies to thinking-capable backends (Qwen3 via Ollama, Claude via
    # the API); ignored by models that don't think.
    think: str = "off"                        # off | auto | on (off keeps simple tasks fast)
    max_output_tokens: int = 32000            # main answer generation
    meta_output_tokens: int = 16000           # optimizer / evolve calls
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
    # --- the benchmark as a PROCESS, not a constant (ag/benchgen.py) ---------
    # 14 fixed tasks cap evolution at "passes those 14", and a fixed suite is
    # memorizable — a LoRA run over AG's history can learn the answers without the
    # capability. Generated tasks are unbounded, reproducible from a seed, and split
    # into train/validation so adoption can be confirmed on items evolution never
    # optimised against. A curated state/bench/tasks.jsonl still overrides this.
    bench_generated: bool = False               # use generated tasks as the suite
    bench_generated_n: int = 40                 # how many tasks per suite
    bench_seed: int = 1337                      # suite seed (same seed = same suite)
    bench_tier: int = 2                         # 1 easy | 2 normal | 3 hard | 0 mixed
    bench_validate: bool = True                 # re-check an adopted edit on held-out tasks
    # Tasks are independent, so they can run concurrently. 1 by default: a single
    # local GPU serialises anyway and concurrency would only add queueing. Raise it
    # for an API backend, where it is close to a linear wall-clock win on the
    # dominant cost of an evolve cycle.
    bench_workers: int = 1
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
    # --- Run budget (ag/budget.py) -------------------------------------------
    # An explicit ceiling on what ONE run may spend, enforced by the loop, not the
    # model. All default 0 = unlimited (existing behaviour); set them for unattended
    # or autonomous runs, where an unbounded loop is the one guaranteed way to burn
    # GPU-hours or API money on a task that will never converge.
    budget_model_calls: int = 0                # max model calls per run (0 = unlimited)
    budget_tool_calls: int = 0                 # max tool invocations per run
    budget_tokens: int = 0                     # max in+out tokens per run
    budget_wall_s: float = 0.0                 # max wall seconds per run
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
    # Vetted GitHub as evolutionary options: when evolving, inject these reference
    # files (adapted from proven repos) into the proposer's briefing so a self-edit can
    # draw on battle-tested implementations. Each entry: {repo, path, note}. Reference
    # data only — evolve still edits ONLY evolvable_paths and passes both gates. Repos
    # must be on github_allowlist. Off unless refs are configured.
    evolve_use_github: bool = False
    evolve_github_refs: List[dict] = field(default_factory=list)
    # --- LoRA fine-tuning (optional; ag/lora.py) ----------------------------
    # Weight-level self-improvement of a LOCAL model via QLoRA. Heavy, opt-in, and
    # kept out of the core import path so the base tool stays stdlib-portable. Needs
    # the extras in requirements-lora.txt (torch/transformers/peft/datasets/bitsandbytes)
    # and an NVIDIA GPU. On an 8GB card this targets a 7-8B base in 4-bit ("stretch").
    lora_base_model: str = "OBLITERATUS/Qwen2.5-Coder-7B-Instruct-OBLITERATED"  # base to specialize (selectable)
    lora_use_unsloth: bool = True              # use Unsloth when installed (less VRAM, faster);
                                               # falls back to transformers+peft when absent
    lora_4bit: bool = True                     # QLoRA 4-bit base (required on small VRAM)
    lora_grad_checkpointing: bool = True       # recompute activations -> big VRAM saving
    lora_optimizer: str = "paged_adamw_8bit"   # paged 8-bit: small state, spills to RAM
    lora_r: int = 16                           # LoRA rank
    lora_alpha: int = 32                       # LoRA alpha
    # 0, not the customary 0.05: Unsloth silently falls back off its fused LoRA kernels
    # whenever dropout is non-zero, and on short runs (tens of steps) the regularisation
    # that buys is negligible next to the throughput it costs. Raise it only for long
    # runs on a large dataset, where overfitting is a real risk — the trade is reported
    # at train time so it is never paid by accident.
    lora_dropout: float = 0.0
    lora_epochs: float = 1.0
    lora_lr: float = 2e-4
    lora_warmup_ratio: float = 0.03            # ease into the LR; short runs need it most
    lora_lr_scheduler: str = "cosine"          # decay after warmup (standard QLoRA)
    # Working budget for Unsloth's fused cross-entropy on a small card, where its own
    # free-VRAM probe reads ~0 mid-run and aborts training. 0 disables the pin.
    lora_ce_target_gb: float = 0.5
    lora_max_seq: int = 1024                   # detection lowers this on tight VRAM
    lora_batch_size: int = 1
    lora_grad_accum: int = 8                   # effective batch without the memory cost
    lora_min_examples: int = 16                # refuse to train on too little data
    lora_teacher_backend: str = "anthropic"    # backend that generates the teacher set
    # Closing the loop: after training, merge the adapter into the base and convert to
    # GGUF so the fine-tuned model becomes a selectable Ollama backend.
    lora_gguf_quant: str = "q4_k_m"            # quantization for the exported GGUF
    gguf_convert_script: str = ""              # path to llama.cpp convert_hf_to_gguf.py (auto-found if empty)
    # Inject a short briefing about THIS machine (OS, shell, package manager, path
    # conventions) into the executor prompt. Cheap, and it removes a whole class of
    # confidently-wrong platform-specific answers.
    os_context: bool = True
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
        "semantic": 0.45, "keyword": 0.18, "recency": 0.12, "importance": 0.10,
        "confidence": 0.15,
    })
    memory_recency_halflife_days: float = 30.0  # recency decay half-life
    memory_merge_threshold: float = 0.92        # cosine >= this => near-duplicate, merged
    # --- Veracity: what AG believes, as opposed to what it merely heard -----
    # Belief is seeded from a memory's origin (user > own observation > distillation >
    # inference > web) and rises only on INDEPENDENT corroboration, so repetition can
    # never manufacture confidence. Below the floor a memory is not recalled at all;
    # between floor and trust it is recalled explicitly as an unconfirmed hypothesis.
    memory_confidence_floor: float = 0.2        # below this, never surfaced
    memory_trust_threshold: float = 0.65        # at/above this, stated as established
    memory_confidence_halflife_days: float = 180.0  # volatile beliefs decay to "unknown"
    memory_contradiction_threshold: float = 0.72    # similarity at which two claims clash
    memory_caps: dict = field(default_factory=lambda: {  # per-layer retention caps
        "episodic": 2000, "semantic": 1000, "procedural": 500,
    })
    memory_reflect: bool = True                 # distill episodes -> facts/procedures
    memory_reflect_every: int = 10              # run reflection every N auto-memory writes
    # --- Auto-inject discipline (B1): keep the durable block from crowding the window.
    # Only what AG can actually stand on is auto-injected; unconfirmed hypotheses are
    # capped hard and otherwise reached on demand via the recall tool.
    memory_inject_k: int = 4                     # max durable facts recalled into context
    memory_inject_max_reported: int = 2          # max unconfirmed hypotheses auto-injected
    memory_inject_min_relevance: float = 0.55    # auto-inject only topically-relevant memory
                                                 # (shared word, or semantic cosine >= this)
    # --- Working memory (A3): per-session buffer for conversational coherence -----
    # Summary + verbatim recent turns + a pinned decision ledger, scoped to one session
    # and kept structurally apart from the durable store (see ag/memory/working.py).
    working_memory: bool = True                  # use the per-session working buffer
    working_recent_turns: int = 6               # verbatim exchanges kept in the tail
    working_summary_chars: int = 700            # rendered rolling-summary budget
    working_idle_reset_min: int = 45            # CLI: silence longer than this -> new session
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
