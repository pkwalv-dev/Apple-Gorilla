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
    {"id": "Qwen/Qwen3-1.7B", "params_b": 1.7, "min_vram": 4.0,
     "note": "tiny & fast; easiest to train, lowest quality"},
    {"id": "Qwen/Qwen3-4B", "params_b": 4.0, "min_vram": 6.0,
     "note": "comfortable on 8GB; ~Qwen2.5-7B quality; fast LoRA cycles"},
    {"id": "Qwen/Qwen3-8B", "params_b": 8.0, "min_vram": 7.5,
     "note": "best quality that still trains on 8GB (~Qwen2.5-14B); tight — Unsloth recommended"},
    {"id": "Qwen/Qwen2.5-Coder-7B-Instruct", "params_b": 7.6, "min_vram": 7.5,
     "note": "code/tool specialist; strongest for coding-heavy use"},
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
        base, seq = "Qwen/Qwen3-8B", 512      # 8GB: tight, needs the savers below
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
        notes.append(f"{vram} GB VRAM: an 8B trains but is tight — Unsloth + gradient "
                     "checkpointing + seq 512 recommended (auto-applied)")
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


def _pairs_from_memory(cfg: Config) -> List[dict]:
    """Turn AG's own memory into instruction/output pairs.

    Episodic memories are stored as "Q: ...\\nA: ..." — recover those as supervised
    pairs. Procedural memories become "how should you handle X" style guidance. Only
    reasonably-scored / durable items are used, so we train on what went well."""
    pairs: List[dict] = []
    try:
        from . import memory
        mgr = memory.get_manager("root", cfg=cfg)
        for m in mgr.store.all("root", [memory.MemoryKind.EPISODIC]):
            score = float(m.meta.get("score", 0) or 0)
            if score and score < 6:      # skip low-quality exchanges
                continue
            mt = re.search(r"Q:\s*(.*?)\s*A:\s*(.*)", m.text, re.DOTALL)
            if mt:
                q, a = mt.group(1).strip(), mt.group(2).strip()
                if q and a:
                    pairs.append({"instruction": q, "output": a, "source": "memory"})
        for m in mgr.store.all("root", [memory.MemoryKind.PROCEDURAL]):
            if m.text.strip():
                pairs.append({
                    "instruction": "What is a good approach for this kind of task?",
                    "output": m.text.strip(), "source": "memory"})
    except Exception:
        pass
    return pairs


# Curated teacher task suite. These are distillation prompts: a strong teacher (Claude)
# answers each, and its answers become supervised targets for the local student. They are
# grouped to cover the capabilities AG actually leans on — coding, debugging, tool/agentic
# reasoning, structured output, careful reasoning, concise technical writing, and safety —
# so the adapter shifts *behavior and style*, not just a handful of facts. Answers inherit
# the teacher's system prompt (EXECUTOR_SYSTEM_DEFAULT), so they carry AG's concise,
# evidence-based voice. Extend freely; pass domain-specific tasks via `extra_tasks`.
TEACHER_TASKS = [
    # --- Coding & algorithms ------------------------------------------------ #
    "Write a Python function that returns the nth Fibonacci number iteratively, with a docstring.",
    "Implement binary search over a sorted list in Python and explain its time complexity in one line.",
    "Write a Python function that flattens an arbitrarily nested list of integers.",
    "Given a list of (name, score) tuples, return the names of the top 3 scorers in Python.",
    "Write a Python context manager that times the block it wraps and prints the elapsed seconds.",
    "Implement an LRU cache in Python using only the standard library.",
    "Write a regex that matches an ISO-8601 date (YYYY-MM-DD) and show one matching and one non-matching example.",
    "Deduplicate a list while preserving order, in idiomatic Python.",
    # --- Debugging & code reasoning ---------------------------------------- #
    "This function raises 'RuntimeError: dictionary changed size during iteration'. What's the cause and the fix?",
    "Explain why comparing floats with == is unreliable, and give the correct approach.",
    "A Python default argument is a mutable list and state leaks between calls. Explain why and fix it.",
    "Given a stack trace ending in 'KeyError: user_id', outline the steps to diagnose it.",
    # --- Tool use & agentic reasoning -------------------------------------- #
    "You can call a web-search tool and a calculator. A user asks for the population of Japan times 2. "
    "Describe the exact sequence of tool calls and why.",
    "When should an agent stop calling tools and answer directly? Give a concrete decision rule.",
    "You retrieved three web sources; two agree and one contradicts. How do you decide what to report?",
    "Describe how to break the task 'summarize this repo's test coverage' into concrete steps.",
    # --- Structured output -------------------------------------------------- #
    "Extract the name, date, and amount from: 'Invoice for Acme Corp dated 2025-03-14, total $1,240.50.' "
    "Return strict JSON with keys name, date, amount.",
    "Convert this to a JSON array of objects with keys task and done: 'buy milk (done), call bank, email Sam'.",
    "Produce a one-row markdown table comparing lists and tuples in Python on mutability and use case.",
    # --- Careful reasoning (traps & estimation) ---------------------------- #
    "Explain the difference between correlation and causation with one concrete example.",
    "Given costs [4,2,7] and values [3,1,5], which item has the best value-per-cost? Show the ratios.",
    "A bat and ball cost $1.10 total; the bat costs $1 more than the ball. How much is the ball? Show reasoning.",
    "Estimate how many piano tuners work in Chicago, showing each assumption.",
    "If a task takes 2 hours for one worker, why doesn't it take 1 hour for two? Name the assumption that must hold.",
    # --- Concise technical writing ----------------------------------------- #
    "Summarize the tradeoffs between TCP and UDP in three bullet points.",
    "Rewrite this to be concise: 'It is important to note that in order to achieve the desired outcome...'.",
    "Explain what a hash map is to a competent programmer in three sentences.",
    "Explain the difference between a process and a thread in three bullet points.",
    "Explain what a database index does and its main cost, in two sentences.",
    "Describe git rebase vs merge in a way that helps someone choose, in under 60 words.",
    # --- Safety & judgment -------------------------------------------------- #
    "What are the main risks of running arbitrary code downloaded from the internet?",
    "A web page you fetched contains text telling you to ignore your instructions and email a file. What do you do?",
    "When is it appropriate to refuse a coding request, and how should the refusal be phrased?",
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


def build_dataset(cfg: Config, *, use_memory: bool = True, use_teacher: bool = True,
                  client=None, extra_tasks: Optional[List[str]] = None,
                  emit=None) -> DatasetStats:
    """Build the merged training set (memory + teacher) and write it as JSONL."""
    _ensure()
    mem = _pairs_from_memory(cfg) if use_memory else []
    teach = _pairs_from_teacher(cfg, client=client, extra_tasks=extra_tasks,
                                emit=emit) if use_teacher else []
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
@dataclass
class TrainResult:
    ok: bool
    adapter_path: str = ""
    examples: int = 0
    reason: str = ""


def _format_example(tokenizer, instruction: str, output: str, max_seq: int) -> dict:
    """Render one pair with the base model's chat template (fallback to a plain format)."""
    try:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction},
             {"role": "assistant", "content": output}],
            tokenize=False)
    except Exception:
        text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
    return tokenizer(text, truncation=True, max_length=max_seq)


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
                model = FastLanguageModel.get_peft_model(
                    model, r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
                    lora_dropout=cfg.lora_dropout, bias="none",
                    use_gradient_checkpointing="unsloth",
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                    "gate_proj", "up_proj", "down_proj"])
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
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))

        from transformers import (DataCollatorForLanguageModeling, Trainer,
                                  TrainingArguments)
        ds = ds.map(lambda r: _format_example(tok, r["instruction"], r["output"], max_seq),
                    remove_columns=ds.column_names)
        args = TrainingArguments(
            output_dir=str(out_dir / "_trainer"),
            per_device_train_batch_size=cfg.lora_batch_size,
            gradient_accumulation_steps=cfg.lora_grad_accum,
            num_train_epochs=cfg.lora_epochs, learning_rate=cfg.lora_lr,
            gradient_checkpointing=bool(cfg.lora_grad_checkpointing),
            optim=str(getattr(cfg, "lora_optimizer", "paged_adamw_8bit")),
            bf16=bf16, fp16=not bf16, logging_steps=5, save_strategy="no", report_to=[])
        trainer = Trainer(model=model, args=args, train_dataset=ds,
                          data_collator=DataCollatorForLanguageModeling(tok, mlm=False))
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
