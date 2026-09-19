"""LoRA / QLoRA fine-tuning — weight-level self-improvement of a LOCAL model.

This is the one part of AG that changes a model's *weights*, not its scaffolding. It
specializes a local base model (never Claude — you have no weight access to it) on a
dataset distilled from AG's own experience plus a Claude-generated teacher set, using
QLoRA so a 7-8B model is trainable on a small GPU (e.g. an 8GB RTX 4060 in 4-bit).

Deliberately OPTIONAL and out of the core import path:
- The heavy deps (torch, transformers, peft, datasets, bitsandbytes) are imported
  lazily inside `train()`, and listed in requirements-lora.txt — so the base tool stays
  pure-Python and portable, and `bundle --check` keeps passing.
- Building the dataset needs none of them (stdlib + an optional model client), so you
  can prepare data anywhere and train only where a GPU exists.

Flow: build_dataset() -> train() -> a LoRA adapter under state/lora/adapters/<ts>/.
Nothing here trains automatically or in the background; training is always explicit.
"""
from __future__ import annotations

import json
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .config import ROOT, STATE_DIR, Config

LORA_DIR = STATE_DIR / "lora"
DATASET_FILE = LORA_DIR / "dataset.jsonl"
ADAPTERS_DIR = LORA_DIR / "adapters"

_HEAVY_DEPS = ("torch", "transformers", "peft", "datasets", "bitsandbytes")

# Curated QLoRA-trainable bases, smallest first. `params_b` is billions of params;
# `min_vram` is the rough VRAM (GB) to QLoRA-train it in 4-bit with the hardened config
# (gradient checkpointing + paged optimizer, seq ~512-1024). `note` is shown in the UI.
BASES = [
    {"id": "OBLITERATUS/Qwen2.5-Coder-7B-Instruct-OBLITERATED", "params_b": 7.6,
     "min_vram": 7.5,
     "note": "AG default: abliterated (uncensored) coder base; strongest for coding/tool "
             "use and won't refuse tasks. Public full weights."},
    {"id": "Qwen/Qwen3-1.7B", "params_b": 1.7, "min_vram": 4.0,
     "note": "tiny & fast; easiest to train, lowest quality"},
    {"id": "Qwen/Qwen3-4B", "params_b": 4.0, "min_vram": 6.0,
     "note": "comfortable on 8GB; ~Qwen2.5-7B quality; fast LoRA cycles"},
    {"id": "Qwen/Qwen3-8B", "params_b": 8.0, "min_vram": 7.5,
     "note": "best quality that still trains on 8GB (~Qwen2.5-14B); tight — Unsloth recommended"},
    {"id": "meta-llama/Llama-3.1-8B-Instruct", "params_b": 8.0, "min_vram": 7.5,
     "note": "solid general 8B; broadest tooling; gated on HF (needs access)"},
    {"id": "Qwen/Qwen3-14B", "params_b": 14.0, "min_vram": 12.0,
     "note": "inference-grade; won't QLoRA-train on 8GB"},
]


# --------------------------------------------------------------------------- #
# Feasibility / preflight
# --------------------------------------------------------------------------- #
@dataclass
class Feasibility:
    ok: bool
    gpu: str = ""
    vram_gb: Optional[float] = None
    cuda: bool = False
    unsloth: bool = False
    missing_deps: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"ok": self.ok, "gpu": self.gpu, "vram_gb": self.vram_gb,
                "cuda": self.cuda, "unsloth": self.unsloth,
                "missing_deps": self.missing_deps, "notes": self.notes}


def _vram_gb() -> Optional[float]:
    """Total VRAM of the first NVIDIA GPU, in GB (None if no nvidia-smi)."""
    import shutil
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5)
        if out.returncode == 0 and out.stdout.strip():
            return round(int(out.stdout.strip().splitlines()[0]) / 1024, 1)
    except Exception:
        pass
    return None


def _missing_deps() -> List[str]:
    import importlib.util
    return [d for d in _HEAVY_DEPS if importlib.util.find_spec(d) is None]


def _has_unsloth() -> bool:
    import importlib.util
    return importlib.util.find_spec("unsloth") is not None


def available_bases(vram_gb: Optional[float] = None) -> List[dict]:
    """The trainable-base catalog, each annotated with how it fits THIS GPU's VRAM."""
    if vram_gb is None:
        vram_gb = _vram_gb()
    out = []
    for b in BASES:
        fit = "unknown"
        if vram_gb is not None:
            if vram_gb >= b["min_vram"] + 2:
                fit = "fits"
            elif vram_gb >= b["min_vram"]:
                fit = "tight"
            else:
                fit = "won't fit"
        out.append({**b, "fit": fit})
    return out


def recommended_config(vram_gb: Optional[float] = None) -> dict:
    """VRAM-sized training knobs: base suggestion + seq length + memory savers."""
    if vram_gb is None:
        vram_gb = _vram_gb()
    v = vram_gb or 0
    if v >= 16:
        base, seq = "Qwen/Qwen3-8B", 2048
    elif v >= 10:
        base, seq = "Qwen/Qwen3-8B", 1024
    elif v >= 7:
        # 8GB: tight, needs the savers below. 768 rather than 512 because the window
        # has to hold a whole answer — a shorter one does not save memory so much as
        # silently clip the training target, which is the more expensive failure. With
        # Unsloth + 4-bit + gradient checkpointing at batch 1 this fits; if it OOMs,
        # lower lora_max_seq (it is clamped by this value, never raised past it).
        base, seq = "Qwen/Qwen3-8B", 768
    elif v >= 5:
        base, seq = "Qwen/Qwen3-4B", 768
    else:
        base, seq = "Qwen/Qwen3-1.7B", 512
    return {"base": base, "max_seq": seq, "grad_checkpointing": True,
            "optimizer": "paged_adamw_8bit", "load_in_4bit": True,
            "prefer_unsloth": True}


def feasibility(cfg: Optional[Config] = None) -> Feasibility:
    """Report whether a LoRA run can happen on this host, and what's missing."""
    from . import host
    gpu = host._gpu()
    vram = _vram_gb()
    missing = _missing_deps()
    cuda = False
    if "torch" not in missing:
        try:
            import torch  # noqa
            cuda = bool(torch.cuda.is_available())
        except Exception:
            cuda = False
    unsloth = _has_unsloth()
    notes: List[str] = []
    if missing:
        notes.append("install the training extras: pip install -r requirements-lora.txt")
    if not unsloth and not missing:
        notes.append("Unsloth not installed — training will fall back to transformers+peft "
                     "(more VRAM, slower). pip install unsloth for the 8GB-friendly path")
    if vram is None:
        notes.append("no NVIDIA GPU detected — QLoRA needs CUDA; training is CPU-infeasible")
    elif vram < 7:
        notes.append(f"{vram} GB VRAM: use a <=4B base (an 8B won't fit for training)")
    elif vram < 10:
        # Read the real recommendation rather than restating a number that can drift
        # out of step with it — a status line quoting a stale value is a small lie in
        # the one place you go to find out what will happen.
        seq = recommended_config(vram)["max_seq"]
        notes.append(f"{vram} GB VRAM: an 8B trains but is tight — Unsloth + gradient "
                     f"checkpointing + seq {seq} recommended (auto-applied; lower "
                     f"lora_max_seq if training runs out of memory)")
    if "bitsandbytes" not in missing and vram is not None:
        notes.append("bitsandbytes/Unsloth on native Windows can be finicky; if 4-bit "
                     "fails to load, run under WSL2")
    ok = (not missing) and cuda and (vram is not None)
    return Feasibility(ok=ok, gpu=gpu, vram_gb=vram, cuda=cuda, unsloth=unsloth,
                       missing_deps=missing, notes=notes)


# --------------------------------------------------------------------------- #
# Dataset construction (no heavy deps)
# --------------------------------------------------------------------------- #
@dataclass
class DatasetStats:
    total: int
    from_memory: int
    from_teacher: int
    path: str


def _ensure() -> None:
    LORA_DIR.mkdir(parents=True, exist_ok=True)
    ADAPTERS_DIR.mkdir(parents=True, exist_ok=True)


# Identity/self-description questions must NOT train from memory: AG's identity is set
# authoritatively by the system prompt (prompts.EXECUTOR_SYSTEM_DEFAULT). Past episodes
# captured stale/incorrect self-descriptions ("a protocol layer, no persistent memory,
# no file access"); training on them teaches the model to deny its own capabilities and
# fights the system prompt. So we drop identity Q/A from the memory-derived training set.
_IDENTITY_Q_MARKERS = (
    "your name", "who are you", "what are you", "about yourself", "what is apple-gorilla",
    "who is apple-gorilla", "what do you know about yourself", "gotten smarter",
    "gotten any smarter", "are you sentient", "are you conscious", "describe yourself",
)


def _is_identity_question(q: str) -> bool:
    ql = q.lower()
    return any(mk in ql for mk in _IDENTITY_Q_MARKERS)


# A supervised pair has to carry a lesson. Roughly one sentence of answer is the floor:
# below that it is an acknowledgement or a quip, and training on it pushes the model
# toward replying with nothing.
_MIN_PROMPT_CHARS = 8
_MIN_ANSWER_CHARS = 60

_PROC_SEP = " — "
# "For arithmetic, call calc first." / "When the build is red, bisect." — the situation
# a free-text procedure applies to, recoverable from its opening clause.
_PROC_SITUATION = re.compile(r"^(for|when|if|while|during|with)\s+([^,.;]{3,60})[,;]",
                             re.IGNORECASE)


def _procedure_pair(m) -> Optional[dict]:
    """Turn one procedural memory into a pair whose QUESTION is the situation it is for.

    Every procedure used to be filed under the same instruction, "What is a good
    approach for this kind of task?" — one prompt mapped onto a dozen unrelated
    answers, which teaches the model that the question carries no information and that
    any of those answers is an acceptable response to it. A procedure already records
    the situation it applies to; asking about that situation is what makes the pair
    teach something.

    Returns None for a procedure with no recoverable situation: a bare one-liner would
    only reintroduce a generic, colliding instruction.
    """
    text = m.text.strip()
    tags = [t for t in (m.tags or []) if t and t != "skill"]
    if "skill" in (m.tags or []) and tags:
        return {"instruction": f"When should you use the '{tags[0]}' skill, and what "
                               f"does it do?",
                "output": text, "source": "memory"}

    # Reflection's shape: "name — when <situation> — steps" (see reflect._fmt_procedure).
    parts = [p.strip() for p in text.split(_PROC_SEP) if p.strip()]
    idx = next((i for i, p in enumerate(parts) if p.lower().startswith("when ")), -1)
    if idx >= 0:
        situation = parts[idx][len("when "):].strip().rstrip("?.")
        # The steps are the answer. The leading name is a label for the procedure, not
        # something a model should learn to say back, so prefer what follows the
        # situation and fall back to the name only when there are no steps.
        rest = _PROC_SEP.join(parts[idx + 1:]).strip() or _PROC_SEP.join(parts[:idx]).strip()
        if situation and rest:
            return {"instruction": f"What is the best approach when {situation}?",
                    "output": rest, "source": "memory"}

    mt = _PROC_SITUATION.match(text)
    if mt:
        lead, situation = mt.group(1).lower(), mt.group(2).strip()
        ask = "when" if lead in ("when", "if", "while", "during") else "for"
        return {"instruction": f"What is a good approach {ask} {situation}?",
                "output": text, "source": "memory"}

    if len(parts) > 1 and parts[0]:
        return {"instruction": f"What is a good approach for {parts[0].rstrip(':')}?",
                "output": _PROC_SEP.join(parts[1:]), "source": "memory"}
    return None


def _pairs_from_memory(cfg: Config) -> List[dict]:
    """Turn AG's own memory into instruction/output pairs.

    Episodic memories are stored as "Q: ...\\nA: ..." — recover those as supervised
    pairs. Procedural memories become "how should you handle X" style guidance.

    Training is the one place where a memory stops being revisable. Everything else in
    AG can re-weigh a belief later — lower its confidence, mark it disputed, forget it —
    but once a pair is in the dataset it is pressed into the weights, where none of that
    reaches it. So this is the strictest filter in the system, and it is deliberately
    stricter than recall:

    - Only BELIEVED procedures train. Reflection writes procedures as unconfirmed
      hypotheses (~0.45); fine-tuning on those would turn a guess about what works into
      a reflex. A procedure must have been corroborated or verified first.
    - Disputed and superseded memories never train, at any confidence.
    - Web-derived answers never train. They were fine to answer with, attributed and
      caveated, but a model cannot carry the caveat into its weights — it would learn
      the claim and lose the attribution.
    - Self-descriptions never train (see _IDENTITY_Q_MARKERS): AG's identity comes from
      the system prompt, and a fine-tuned self-description fights it forever.
    - Clipped answers never train. An answer cut mid-sentence teaches the model to stop
      mid-sentence, which is a defect no amount of good data elsewhere undoes.
    - Threadbare exchanges never train. Most episodes carry no score at all (the
      pipeline does not score a normal run), so `score` cannot be the quality bar
      without emptying this source entirely. The bar is the content itself: a
      one-line throwaway like "Well, what do you know!" is not a lesson.
    """
    pairs: List[dict] = []
    try:
        from . import memory
        from .memory.manager import LEGACY_EPISODE_CHARS
        mgr = memory.get_manager("root", cfg=cfg)
        for m in mgr.store.all("root", [memory.MemoryKind.EPISODIC]):
            score = float(m.meta.get("score", 0) or 0)
            if score and score < 6:      # an explicit bad score disqualifies
                continue
            if m.meta.get("web_sources"):
                continue                 # untrusted content must not become a weight
            if m.meta.get("truncated"):
                continue                 # recorded as clipped when it was stored
            mt = re.search(r"Q:\s*(.*?)\s*A:\s*(.*)", m.text, re.DOTALL)
            if mt:
                q, a = mt.group(1).strip(), mt.group(2).strip()
                # Episodes stored under the old 400-character cap carry no `truncated`
                # flag; an answer landing exactly on it was almost certainly cut.
                if len(a) == LEGACY_EPISODE_CHARS:
                    continue
                if len(q) < _MIN_PROMPT_CHARS or len(a) < _MIN_ANSWER_CHARS:
                    continue             # too thin to be teaching anything
                # The blunt self-reference filter (question AND answer) — see
                # memory.is_self_reference and _IDENTITY_Q_MARKERS.
                if not (memory.is_self_reference(q) or memory.is_self_reference(a)):
                    pairs.append({"instruction": q, "output": a, "source": "memory"})
        for m in mgr.store.all("root", [memory.MemoryKind.PROCEDURAL]):
            if not m.text.strip():
                continue
            if m.disputed or m.meta.get("superseded_by"):
                continue
            if mgr.belief(m) < mgr.trust_threshold:
                continue                 # an unconfirmed method is not a lesson yet
            pair = _procedure_pair(m)
            if pair:
                pairs.append(pair)
    except Exception:
        pass
    return pairs


# Curated teacher task suite. These are distillation prompts: a strong teacher (Claude)
# answers each, and its answers become supervised targets for the local student. They are
# grouped to cover the capabilities AG actually leans on, with deliberate weight on the
# three foci — tool/file interaction, agentic swarms, and self-improving programs — while
# keeping every other area well represented so nothing is undertrained. Answers inherit the
# teacher's system prompt (EXECUTOR_SYSTEM_DEFAULT), so they carry AG's concise, evidence-
# based voice. Extend freely; pass domain-specific tasks via `extra_tasks`.
TEACHER_TASKS = [
    # --- Coding & algorithms (16) ------------------------------------------ #
    "Write a Python function that returns the nth Fibonacci number iteratively, with a docstring.",
    "Implement binary search over a sorted list in Python and explain its time complexity in one line.",
    "Write a Python function that flattens an arbitrarily nested list of integers.",
    "Given a list of (name, score) tuples, return the names of the top 3 scorers in Python.",
    "Write a Python context manager that times the block it wraps and prints the elapsed seconds.",
    "Implement an LRU cache in Python using only the standard library.",
    "Write a regex that matches an ISO-8601 date (YYYY-MM-DD) and show one matching and one non-matching example.",
    "Deduplicate a list while preserving order, in idiomatic Python.",
    "Write a Python generator that yields lines from a large file without loading it all into memory.",
    "Merge two sorted lists into one sorted list without calling sorted(), in Python.",
    "Write a Python decorator that retries a function up to N times with exponential backoff.",
    "Given a dict of {item: count}, return the item with the highest count, breaking ties alphabetically.",
    "Write a Python function that computes the Levenshtein distance between two strings.",
    "Parse a 'key=value' config string into a dict, ignoring blank lines and '#' comments.",
    "Implement a thread-safe counter in Python.",
    "Write a Python function that groups a list of dicts by a given key.",
    # --- Debugging & code reasoning (10) ----------------------------------- #
    "This function raises 'RuntimeError: dictionary changed size during iteration'. What's the cause and the fix?",
    "Explain why comparing floats with == is unreliable, and give the correct approach.",
    "A Python default argument is a mutable list and state leaks between calls. Explain why and fix it.",
    "Given a stack trace ending in 'KeyError: user_id', outline the steps to diagnose it.",
    "A script works when run directly but fails on import with 'ModuleNotFoundError'. List the likely causes.",
    "Code intermittently deadlocks under threads. Describe how you'd reproduce and locate the cause.",
    "A regex is catastrophically slow on some inputs. Explain the likely cause and the fix.",
    "Explain the difference between '==' and 'is' in Python with an example where they disagree.",
    "A unit test passes alone but fails in the full suite. What are the usual causes?",
    "Memory usage grows unbounded in a long-running loop. Outline how to find the leak.",
    # --- Tool use & file interaction (FOCUS, 28) --------------------------- #
    "You can call a web-search tool and a calculator. A user asks for the population of Japan times 2. "
    "Describe the exact sequence of tool calls and why.",
    "When should an agent stop calling tools and answer directly? Give a concrete decision rule.",
    "You retrieved three web sources; two agree and one contradicts. How do you decide what to report?",
    "Describe how to break the task 'summarize this repo's test coverage' into concrete steps.",
    "A user says 'read the config file and tell me the timeout' but gives no path. Describe how to locate and read it.",
    "You need to edit a file you have never opened. What must you do before editing, and why?",
    "Given tools {list_files, read_file, write_file, run_shell}, describe how to rename every '.txt' to '.md' "
    "in a directory safely.",
    "A task needs a capability you lack (e.g. reading a PDF). Describe how a self-extending agent should acquire it.",
    "You must find where a function 'compute_fitness' is defined across an unknown codebase. Which tool, and how?",
    "Describe the safe file-edit sequence — read, back up, edit, verify — and why the verify step is non-optional.",
    "A user asks you to write to a protected system file. What checks and permissions should gate that first?",
    "You can run shell commands. Describe how to determine whether a program is installed before using it.",
    "A fetch tool returns 3 MB of HTML but you need one fact. Describe how to extract just what's relevant.",
    "You must apply the same change across 40 files. Describe a safe, reviewable approach using tools.",
    "Explain how an agent should choose between doing a task itself in code vs. calling an external tool.",
    "A tool call failed with a permission error. Describe the correct recovery sequence.",
    "You are told 'get the data, wherever it is'. Describe how to search many likely locations by name and content.",
    "Describe how to read only the relevant slice of a 10,000-line log to find an error, without reading it all.",
    "When editing code with a tool, why should you match the surrounding style, and how do you learn it first?",
    "A file path contains spaces and lives on a Windows machine. Describe how to handle it correctly in a shell command.",
    "Describe how to write a file atomically so a crash mid-write can't corrupt it.",
    "You must open a file specified only as 'the newest report in Downloads'. Describe how to resolve and open it.",
    "Given tools to list, read, and write files anywhere on disk, what guardrails prevent destructive mistakes?",
    "Explain how to verify a shell command's effect after running it, rather than assuming success.",
    "A task needs a Python package that isn't installed. Describe how an agent should install it safely and confirm it.",
    "You can spawn a subprocess. Describe how to capture its stdout, stderr, and exit code and interpret them.",
    "Describe how to decide the minimum set of tools needed for a task, and why fewer is better.",
    "You must move a file to a location the user names at runtime. Describe how to validate the destination first.",
    # --- Agentic swarms & orchestration (FOCUS, 22) ------------------------ #
    "Explain the difference between a single agent with tools and a multi-agent swarm. When is a swarm worth it?",
    "Describe how a parent agent should delegate a subtask to a child agent and validate the result.",
    "What context must a parent pass to a spawned subagent so it can work without re-deriving everything?",
    "How should authority and permissions be bounded when a parent agent spawns children?",
    "Describe a master kill switch for an agent swarm: what it must guarantee and how children honor it.",
    "Two agents produce conflicting answers to the same subtask. How should the orchestrator reconcile them?",
    "Explain how to prevent runaway spawning (agents spawning agents) from exhausting resources.",
    "Describe how to assign unique identities and lineage to agents so their work is traceable.",
    "When should work be parallelized across agents vs. done sequentially by one? Give a decision rule.",
    "How should a swarm share findings between agents without corrupting each other's state?",
    "Describe how an orchestrator aggregates partial results from several agents into one coherent answer.",
    "A subagent hangs and stops responding. How should the orchestrator detect and recover?",
    "Explain how to cap the depth of a spawning tree and why unbounded depth is dangerous.",
    "Describe how to give each agent a clear, non-overlapping responsibility to avoid duplicated work.",
    "How can a swarm inherit skills from a parent while still acquiring new ones itself?",
    "Describe how to test a multi-agent workflow deterministically despite agents being nondeterministic.",
    "What are the failure modes of a swarm where every agent has full permissions, and how do you mitigate them?",
    "Explain how an orchestrator should decide how many agents to spawn for a fan-out task.",
    "A subagent's task fails partway. Should the orchestrator retry, reassign, or abort? Give the reasoning.",
    "How should agents report progress so a human supervisor can follow and intervene?",
    "Design a scheme where subagents can request new capabilities but never exceed their parent's authority.",
    "Explain how to keep a swarm's total cost bounded when each agent may call a paid model.",
    # --- Self-improving programs (FOCUS, 22) ------------------------------- #
    "Explain what it means for a program to self-improve, and the difference between changing its scaffolding "
    "vs. its model weights.",
    "Describe a directed-mutation loop for a self-improving agent and why it beats purely random mutation on "
    "limited hardware.",
    "What is a fitness function here, and what properties make a good one for self-improvement?",
    "Explain how episodic, semantic, and procedural memory let a program learn without retraining weights.",
    "Describe a safe self-modification loop: propose, test, adopt or roll back. Why is the test gate essential?",
    "What is LoRA/QLoRA, and why can it specialize a local model but not a closed API model?",
    "Explain teacher-student distillation and how a strong model can train a smaller local one.",
    "Describe how a program can decide *what* to improve about itself, rather than changing things at random.",
    "How should a self-improving system prevent a change that passes its own tests but degrades real behavior?",
    "Explain reflection: how summarizing past runs into durable lessons improves future performance.",
    "Describe the risk of reward hacking in a self-improving loop and one way to guard against it.",
    "Why should a self-improving program keep improvements reversible, and how do snapshots enable that?",
    "Explain how to measure whether a self-modification actually helped, given noisy performance signals.",
    "Describe how a program acquires a brand-new skill at runtime: author it, validate it, register it, reuse it.",
    "What invariants must a self-editing program never break, and how do you enforce them mechanically?",
    "Explain the tradeoff between improving via better prompts/scaffolding vs. fine-tuning weights.",
    "Describe how to build a training set for self-improvement from a program's own successful experiences.",
    "How should a self-improving agent balance exploiting known-good behavior vs. exploring new approaches?",
    "Explain why 'it passed the tests' is necessary but not sufficient evidence that a self-change is an improvement.",
    "Describe how a program can carry its learned memory across different models or machines as it adapts.",
    "What are the dangers of a self-improving system optimizing a proxy metric instead of the true goal?",
    "Explain how to keep a human in the loop for consequential self-modifications without blocking routine ones.",
    # --- Structured output (9) --------------------------------------------- #
    "Extract the name, date, and amount from: 'Invoice for Acme Corp dated 2025-03-14, total $1,240.50.' "
    "Return strict JSON with keys name, date, amount.",
    "Convert this to a JSON array of objects with keys task and done: 'buy milk (done), call bank, email Sam'.",
    "Produce a one-row markdown table comparing lists and tuples in Python on mutability and use case.",
    "Given a paragraph about a person, return JSON with keys name, role, organization; use null for anything absent.",
    "Turn these into a numbered JSON array of {step, action}: 'first clone the repo, then install deps, then run tests'.",
    "Extract all email addresses from a block of text and return them as a deduplicated JSON array.",
    "Given a shell command, return JSON describing its {command, args, effect} without executing it.",
    "Convert a small CSV snippet into a JSON array of objects keyed by the header row.",
    "Summarize an agent run into JSON with keys task, tools_used, outcome, errors.",
    # --- Careful reasoning (12) -------------------------------------------- #
    "Explain the difference between correlation and causation with one concrete example.",
    "Given costs [4,2,7] and values [3,1,5], which item has the best value-per-cost? Show the ratios.",
    "A bat and ball cost $1.10 total; the bat costs $1 more than the ball. How much is the ball? Show reasoning.",
    "Estimate how many piano tuners work in Chicago, showing each assumption.",
    "If a task takes 2 hours for one worker, why doesn't it take 1 hour for two? Name the assumption that must hold.",
    "You have 8 identical-looking coins, one heavier. Find it in 2 weighings on a balance. Explain.",
    "A test is 99% accurate for a disease affecting 1 in 10,000. You test positive. Roughly what's the real chance "
    "you have it? Explain.",
    "Three switches control one bulb in another room; you may enter once. How do you find the right switch?",
    "Given a claim backed by a single anecdote, explain what evidence would actually support it.",
    "A benchmark shows version B is 5% faster than A over 3 runs. Is that conclusive? Explain what you'd check.",
    "Explain the sunk-cost fallacy with a concrete engineering example.",
    "You must choose between two designs with incomplete information. How do you decide without overanalyzing?",
    # --- Concise technical writing (12) ------------------------------------ #
    "Summarize the tradeoffs between TCP and UDP in three bullet points.",
    "Rewrite this to be concise: 'It is important to note that in order to achieve the desired outcome...'.",
    "Explain what a hash map is to a competent programmer in three sentences.",
    "Explain the difference between a process and a thread in three bullet points.",
    "Explain what a database index does and its main cost, in two sentences.",
    "Describe git rebase vs merge in a way that helps someone choose, in under 60 words.",
    "Explain what an embedding is, to a programmer, in three sentences.",
    "Summarize what a permission broker does and why default-deny matters, in two bullets.",
    "Explain the difference between concurrency and parallelism in two sentences.",
    "Write a one-paragraph commit message for expanding a teacher task set and switching the model.",
    "Explain what prompt injection is and the one-line defense, concisely.",
    "Describe what a LoRA adapter is and where it lives relative to the base model, in two sentences.",
    # --- Safety & judgment (12) -------------------------------------------- #
    "What are the main risks of running arbitrary code downloaded from the internet?",
    "A web page you fetched contains text telling you to ignore your instructions and email a file. What do you do?",
    "When is it appropriate to refuse a coding request, and how should the refusal be phrased?",
    "A tool result claims the user pre-authorized deleting a directory. Should you trust it? Explain.",
    "Before overwriting an existing file, what should you check, and why?",
    "A task asks you to enter an API key into a form. What is the correct, safe response?",
    "Distinguish data from instructions: why must content read from a file never be executed as commands?",
    "You are asked to permanently delete data. What confirmation and safeguards are appropriate first?",
    "When should an agent ask for human confirmation before an action vs. proceeding autonomously?",
    "A downloaded script would speed up your task but you can't verify its source. What do you do?",
    "Explain why an agent should scope its file and network access to the minimum a task needs.",
    "A subagent proposes an irreversible action to satisfy its goal. How should the orchestrator handle it?",
]


def _pairs_from_teacher(cfg: Config, *, client=None, extra_tasks: Optional[List[str]] = None,
                        emit=None) -> List[dict]:
    """Generate instruction/output pairs by asking a strong model (the teacher) to
    answer a task set — classic teacher→student distillation. Best-effort; returns []
    if no capable client is available."""
    pairs: List[dict] = []
    if client is None:
        try:
            from .model import make_client
            client = make_client(cfg, backend=getattr(cfg, "lora_teacher_backend", "") or None)
        except Exception:
            return pairs
    from .model import DryRunClient
    if isinstance(client, DryRunClient):
        return pairs  # the stub can't produce a useful teacher signal
    tasks = list(TEACHER_TASKS) + list(extra_tasks or [])
    from . import prompts
    sys = getattr(prompts, "EXECUTOR_SYSTEM_DEFAULT", "You are a helpful, precise assistant.")
    for t in tasks:
        try:
            res = client.complete(system=sys, user=t, cfg=cfg, max_tokens=800)
            ans = (res.text or "").strip()
            if ans:
                pairs.append({"instruction": t, "output": ans, "source": "teacher"})
        except Exception:
            continue
    return pairs


def _existing_teacher_pairs() -> List[dict]:
    """Teacher rows already on disk from an earlier build."""
    if not DATASET_FILE.exists():
        return []
    out: List[dict] = []
    for ln in DATASET_FILE.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            r = json.loads(ln)
        except Exception:
            continue
        if r.get("source") == "teacher" and r.get("instruction") and r.get("output"):
            out.append(r)
    return out


def build_dataset(cfg: Config, *, use_memory: bool = True, use_teacher: bool = True,
                  client=None, extra_tasks: Optional[List[str]] = None,
                  refresh_teacher: bool = False, emit=None) -> DatasetStats:
    """Build the merged training set (memory + teacher) and write it as JSONL.

    Memory rows are always rebuilt — they are free and they are what changes between
    runs. Teacher rows are REUSED from the previous build unless `refresh_teacher` is
    set, because regenerating them means one API call per task against whatever
    `lora_teacher_backend` points at. A routine rebuild should cost nothing and should
    never quietly discard a teacher set that has already been paid for.
    """
    _ensure()
    from .pipeline import _emit
    mem = _pairs_from_memory(cfg) if use_memory else []
    teach: List[dict] = []
    if use_teacher:
        teach = [] if refresh_teacher else _existing_teacher_pairs()
        if teach:
            _emit(emit, "lora", f"reusing {len(teach)} teacher pair(s) from the previous "
                  f"build (pass refresh to regenerate)", level="info")
        else:
            teach = _pairs_from_teacher(cfg, client=client, extra_tasks=extra_tasks,
                                        emit=emit)
    # De-dup on instruction+output.
    seen = set()
    rows = []
    for r in mem + teach:
        key = (r["instruction"], r["output"])
        if key in seen:
            continue
        seen.add(key)
        rows.append(r)
    DATASET_FILE.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return DatasetStats(total=len(rows), from_memory=len(mem), from_teacher=len(teach),
                        path=str(DATASET_FILE))


def dataset_size() -> int:
    if not DATASET_FILE.exists():
        return 0
    return sum(1 for ln in DATASET_FILE.read_text(encoding="utf-8").splitlines() if ln.strip())


def list_adapters() -> List[dict]:
    if not ADAPTERS_DIR.exists():
        return []
    out = []
    for d in sorted(ADAPTERS_DIR.iterdir(), reverse=True):
        if d.is_dir():
            meta = d / "ag_meta.json"
            info = {"id": d.name, "path": str(d)}
            if meta.exists():
                try:
                    info.update(json.loads(meta.read_text(encoding="utf-8")))
                except Exception:
                    pass
            out.append(info)
    return out


# --------------------------------------------------------------------------- #
# Training (heavy deps, lazy-imported)
# --------------------------------------------------------------------------- #
# Attention AND MLP projections. Both training paths use this same list so that
# falling back from Unsloth to transformers+peft changes only the speed and memory
# profile, never which model you end up with — an adapter that touches attention only
# is a materially different fine-tune, and silently getting one depending on whether
# an optional package happened to be installed is not a fallback, it is a coin toss.
LORA_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj")


@dataclass
class TrainResult:
    ok: bool
    adapter_path: str = ""
    examples: int = 0
    reason: str = ""


def _warmup_steps(cfg: Config, n_examples: int) -> int:
    """How many steps to ease the learning rate in over.

    Derived from the run's real length — examples, effective batch, epochs — so the
    same setting behaves sensibly whether the dataset is 143 examples or 14,000. Short
    runs get a floor of 2: warming up over "0.4 of a step" is the same as not warming
    up, which is precisely when the cold first update hurts most.
    """
    per_step = max(1, int(cfg.lora_batch_size) * int(cfg.lora_grad_accum))
    total = max(1, int(round((n_examples / per_step) * float(cfg.lora_epochs))))
    ratio = float(getattr(cfg, "lora_warmup_ratio", 0.03) or 0.0)
    if ratio <= 0:
        return 0
    return max(2, min(int(round(total * ratio)) or 2, max(1, total // 4)))


def _format_example(tokenizer, instruction: str, output: str, max_seq: int) -> dict:
    """Tokenize one pair, MASKING the prompt so training loss is computed on the answer
    tokens only (label -100 = "ignore"). Without this the model also spends gradient
    learning to predict the question, which dilutes answer quality.

    Also reports whether the pair FIT. An example longer than the window is silently
    cut here, and the cut answer then becomes the training target — the model learns to
    stop mid-sentence. The caller drops anything marked incomplete rather than teaching
    it a truncated answer."""
    try:
        full = tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction},
             {"role": "assistant", "content": output}],
            tokenize=False)
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}],
            tokenize=False, add_generation_prompt=True)
    except Exception:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
        full = prompt + output
    enc = tokenizer(full, truncation=True, max_length=max_seq)
    enc["complete"] = len(tokenizer(full)["input_ids"]) <= max_seq
    n_prompt = min(len(tokenizer(prompt, truncation=True, max_length=max_seq)["input_ids"]),
                   len(enc["input_ids"]))
    labels = list(enc["input_ids"])
    for i in range(n_prompt):
        labels[i] = -100
    if all(tok_id == -100 for tok_id in labels):  # answer truncated away — avoid a NaN row
        labels = list(enc["input_ids"])
    enc["labels"] = labels
    return enc


def train(cfg: Config, *, emit=None) -> TrainResult:
    """Run one QLoRA fine-tune on the built dataset. Requires the training extras and
    a CUDA GPU. Best-effort: returns a TrainResult with a clear reason on any preflight
    failure rather than raising into the caller."""
    from .pipeline import _emit
    feas = feasibility(cfg)
    if feas.missing_deps:
        return TrainResult(False, reason="missing training extras: "
                           + ", ".join(feas.missing_deps)
                           + " (pip install -r requirements-lora.txt)")
    if not feas.cuda:
        return TrainResult(False, reason="no CUDA GPU available — QLoRA needs one")
    n = dataset_size()
    if n < int(getattr(cfg, "lora_min_examples", 16)):
        return TrainResult(False, examples=n,
                           reason=f"dataset too small ({n}); build more first "
                                  f"(need >= {cfg.lora_min_examples})")
    _ensure()
    # Auto-size the sequence length down on tight VRAM (an 8GB card can't hold seq 1024
    # activations for an 8B); never exceed the configured cap.
    rec = recommended_config(feas.vram_gb)
    max_seq = min(int(cfg.lora_max_seq), int(rec["max_seq"]))
    use_unsloth = bool(getattr(cfg, "lora_use_unsloth", True)) and _has_unsloth()
    out_dir = ADAPTERS_DIR / time.strftime("%Y%m%d-%H%M%S")
    _emit(emit, "lora", f"loading {cfg.lora_base_model} (4-bit, seq {max_seq}) via "
          f"{'Unsloth' if use_unsloth else 'transformers+peft'}", level="tool")
    try:
        import torch
        bf16 = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
        from datasets import load_dataset
        ds = load_dataset("json", data_files=str(DATASET_FILE), split="train")

        if use_unsloth:
            # Unsloth: same LoRA/QLoRA result, ~half the VRAM and faster — the path that
            # makes an 8B fit on 8GB. Falls through to transformers+peft if it errors.
            try:
                from unsloth import FastLanguageModel
                model, tok = FastLanguageModel.from_pretrained(
                    model_name=cfg.lora_base_model, max_seq_length=max_seq,
                    load_in_4bit=bool(cfg.lora_4bit), dtype=None)
                # Unsloth drops its fused LoRA kernels the moment dropout is non-zero,
                # and says so only in a line buried in its banner. Say it here, where
                # the cost is being incurred, so a slower run is always a choice.
                if float(cfg.lora_dropout) > 0:
                    _emit(emit, "lora", f"lora_dropout={cfg.lora_dropout} disables "
                          f"Unsloth's fused LoRA kernels — steps will be slower; set "
                          f"lora_dropout to 0 unless the run is long enough for the "
                          f"regularisation to matter", level="info")
                model = FastLanguageModel.get_peft_model(
                    model, r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
                    lora_dropout=cfg.lora_dropout, bias="none",
                    use_gradient_checkpointing="unsloth",
                    target_modules=list(LORA_TARGET_MODULES))
            except Exception as ue:
                _emit(emit, "lora", f"Unsloth path failed ({ue}); falling back to "
                      "transformers+peft", level="info")
                use_unsloth = False

        if not use_unsloth:
            from peft import (LoraConfig, get_peft_model,
                              prepare_model_for_kbit_training)
            from transformers import (AutoModelForCausalLM, AutoTokenizer,
                                      BitsAndBytesConfig)
            tok = AutoTokenizer.from_pretrained(cfg.lora_base_model, use_fast=True)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            quant = None
            if cfg.lora_4bit:
                quant = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=(torch.bfloat16 if bf16 else torch.float16),
                    bnb_4bit_use_double_quant=True)
            model = AutoModelForCausalLM.from_pretrained(
                cfg.lora_base_model, quantization_config=quant, device_map="auto",
                torch_dtype=(torch.bfloat16 if bf16 else torch.float16))
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=bool(cfg.lora_grad_checkpointing))
            model = get_peft_model(model, LoraConfig(
                r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                bias="none", task_type="CAUSAL_LM",
                target_modules=list(LORA_TARGET_MODULES)))

        from transformers import (DataCollatorForSeq2Seq, Trainer,
                                  TrainingArguments)
        ds = ds.map(lambda r: _format_example(tok, r["instruction"], r["output"], max_seq),
                    remove_columns=ds.column_names)
        # Refuse to train on anything the window would clip. Fewer complete examples
        # beat more truncated ones: a clipped target teaches the model to stop early,
        # and no amount of good data elsewhere undoes that.
        fitted = ds.filter(lambda r: r["complete"])
        dropped = len(ds) - len(fitted)
        if dropped:
            _emit(emit, "lora", f"dropped {dropped} of {len(ds)} example(s) longer than "
                  f"the {max_seq}-token window (raise lora_max_seq to keep them)",
                  level="info")
        ds = fitted.remove_columns(["complete"])
        n = len(ds)
        if n < int(getattr(cfg, "lora_min_examples", 16)):
            return TrainResult(False, examples=n,
                               reason=f"only {n} example(s) fit the {max_seq}-token "
                                      f"window (need >= {cfg.lora_min_examples}); "
                                      f"raise lora_max_seq or build more data")
        args = TrainingArguments(
            output_dir=str(out_dir / "_trainer"),
            per_device_train_batch_size=cfg.lora_batch_size,
            gradient_accumulation_steps=cfg.lora_grad_accum,
            num_train_epochs=cfg.lora_epochs, learning_rate=cfg.lora_lr,
            # A run this short (a few dozen optimizer steps) spends a large share of
            # its budget in the first few updates, where a cold full-rate step does the
            # most damage to what the base model already knows. Warm up into it, then
            # decay — the standard QLoRA schedule, and the cheapest guard there is
            # against trading general knowledge for a small set of new habits.
            # Expressed in steps, not a ratio: transformers deprecated warmup_ratio,
            # and a ratio of a very short run rounds down to almost no warmup at all.
            warmup_steps=_warmup_steps(cfg, n),
            lr_scheduler_type=str(getattr(cfg, "lora_lr_scheduler", "cosine")),
            gradient_checkpointing=bool(cfg.lora_grad_checkpointing),
            optim=str(getattr(cfg, "lora_optimizer", "paged_adamw_8bit")),
            bf16=bf16, fp16=not bf16, logging_steps=5, save_strategy="no", report_to=[])
        trainer = Trainer(model=model, args=args, train_dataset=ds,
                          data_collator=DataCollatorForSeq2Seq(tok, padding=True,
                                                               label_pad_token_id=-100))
        _emit(emit, "lora", f"training on {n} examples ({cfg.lora_epochs} epoch(s))",
              level="tool")
        trainer.train()
        out_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(out_dir))
        tok.save_pretrained(str(out_dir))
        (out_dir / "ag_meta.json").write_text(json.dumps({
            "base": cfg.lora_base_model, "examples": n, "r": cfg.lora_r,
            "max_seq": max_seq, "unsloth": use_unsloth,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=2), encoding="utf-8")
        _emit(emit, "lora", f"adapter saved to {out_dir}", level="result")
        return TrainResult(True, adapter_path=str(out_dir), examples=n,
                           reason="training complete")
    except Exception as e:  # pragma: no cover - depends on GPU/deps at runtime
        return TrainResult(False, reason=f"training failed: {e}")


# --------------------------------------------------------------------------- #
# Close the loop: merge adapter -> GGUF -> a selectable Ollama model
# --------------------------------------------------------------------------- #
@dataclass
class MergeResult:
    ok: bool
    ollama_model: str = ""
    gguf_path: str = ""
    reason: str = ""


def _find_gguf_converter(cfg: Config) -> Optional[str]:
    """Locate llama.cpp's convert_hf_to_gguf.py — from config, PATH, or common spots."""
    import shutil
    cand = getattr(cfg, "gguf_convert_script", "") or ""
    if cand and Path(cand).exists():
        return cand
    for name in ("convert_hf_to_gguf.py", "convert-hf-to-gguf.py"):
        found = shutil.which(name)
        if found:
            return found
    for base in (Path.home() / "llama.cpp", Path.home() / "code" / "llama.cpp",
                 ROOT.parent / "llama.cpp"):
        p = base / "convert_hf_to_gguf.py"
        if p.exists():
            return str(p)
    return None


def merge_feasibility(cfg: Config) -> dict:
    """What's needed to close the loop (merge -> GGUF -> Ollama), and what's missing."""
    import shutil
    missing = [d for d in ("torch", "transformers", "peft") if _missing(d)]
    conv = _find_gguf_converter(cfg)
    ollama = bool(shutil.which("ollama"))
    ok = (not missing) and bool(conv) and ollama
    notes = []
    if missing:
        notes.append("needs the training extras (torch/transformers/peft)")
    if not conv:
        notes.append("llama.cpp convert_hf_to_gguf.py not found — clone github.com/"
                     "ggerganov/llama.cpp and set config.gguf_convert_script")
    if not ollama:
        notes.append("ollama not on PATH")
    return {"ok": ok, "converter": conv or "", "ollama": ollama,
            "missing_deps": missing, "notes": notes}


def _missing(mod: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(mod) is None


def merge_to_gguf(cfg: Config, adapter_id: str, *, emit=None) -> MergeResult:
    """Merge a trained adapter into its base, convert to GGUF, and register it with
    Ollama as a new selectable model. Best-effort; returns a clear reason on failure."""
    from .pipeline import _emit
    feas = merge_feasibility(cfg)
    if not feas["ok"]:
        return MergeResult(False, reason="; ".join(feas["notes"]) or "not ready")
    adir = ADAPTERS_DIR / adapter_id
    if not adir.exists():
        return MergeResult(False, reason=f"adapter not found: {adapter_id}")
    try:
        import shutil
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        base = cfg.lora_base_model
        try:
            meta = json.loads((adir / "ag_meta.json").read_text(encoding="utf-8"))
            base = meta.get("base", base)
        except Exception:
            pass
        _emit(emit, "lora", f"merging adapter into {base}", level="tool")
        model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float16,
                                                     device_map="cpu")
        model = PeftModel.from_pretrained(model, str(adir))
        model = model.merge_and_unload()
        merged = adir / "merged"
        merged.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(merged))
        AutoTokenizer.from_pretrained(base).save_pretrained(str(merged))

        conv = _find_gguf_converter(cfg)
        gguf = adir / f"model.{cfg.lora_gguf_quant}.gguf"
        _emit(emit, "lora", "converting merged model to GGUF", level="tool")
        import sys as _sys
        r = subprocess.run([_sys.executable, conv, str(merged), "--outfile", str(gguf),
                            "--outtype", cfg.lora_gguf_quant],
                           capture_output=True, text=True, timeout=3600)
        if r.returncode != 0 or not gguf.exists():
            # Some converter versions don't quantize; fall back to f16 then quantize.
            gguf_f16 = adir / "model.f16.gguf"
            r2 = subprocess.run([_sys.executable, conv, str(merged), "--outfile",
                                 str(gguf_f16), "--outtype", "f16"],
                                capture_output=True, text=True, timeout=3600)
            if r2.returncode != 0 or not gguf_f16.exists():
                return MergeResult(False, reason=f"GGUF conversion failed: "
                                   f"{(r.stderr or r2.stderr)[-300:]}")
            q = shutil.which("llama-quantize") or shutil.which("quantize")
            if q:
                subprocess.run([q, str(gguf_f16), str(gguf), cfg.lora_gguf_quant],
                               capture_output=True, text=True, timeout=1800)
            gguf = gguf if gguf.exists() else gguf_f16

        name = f"ag-{_slug_model(base)}-{adapter_id}"
        modelfile = adir / "Modelfile"
        modelfile.write_text(f"FROM {gguf.name}\n", encoding="utf-8")
        _emit(emit, "lora", f"registering Ollama model {name}", level="tool")
        rc = subprocess.run(["ollama", "create", name, "-f", str(modelfile)],
                            capture_output=True, text=True, timeout=1800, cwd=str(adir))
        if rc.returncode != 0:
            return MergeResult(False, gguf_path=str(gguf),
                               reason=f"ollama create failed: {rc.stderr[-300:]}")
        _emit(emit, "lora", f"Ollama model ready: {name} "
              f"(select it as your backend)", level="result")
        return MergeResult(True, ollama_model=name, gguf_path=str(gguf),
                           reason=f"registered Ollama model '{name}'")
    except Exception as e:  # pragma: no cover - env-dependent
        return MergeResult(False, reason=f"merge failed: {e}")


def _slug_model(hf_id: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", hf_id.split("/")[-1].lower()).strip("-")
