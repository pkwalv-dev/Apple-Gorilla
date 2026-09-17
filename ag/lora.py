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

from .config import STATE_DIR, Config

LORA_DIR = STATE_DIR / "lora"
DATASET_FILE = LORA_DIR / "dataset.jsonl"
ADAPTERS_DIR = LORA_DIR / "adapters"

_HEAVY_DEPS = ("torch", "transformers", "peft", "datasets", "bitsandbytes")


# --------------------------------------------------------------------------- #
# Feasibility / preflight
# --------------------------------------------------------------------------- #
@dataclass
class Feasibility:
    ok: bool
    gpu: str = ""
    vram_gb: Optional[float] = None
    cuda: bool = False
    missing_deps: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"ok": self.ok, "gpu": self.gpu, "vram_gb": self.vram_gb,
                "cuda": self.cuda, "missing_deps": self.missing_deps, "notes": self.notes}


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
    notes: List[str] = []
    if missing:
        notes.append("install the training extras: pip install -r requirements-lora.txt")
    if vram is None:
        notes.append("no NVIDIA GPU detected — QLoRA needs CUDA; training is CPU-infeasible")
    elif vram < 7.5:
        notes.append(f"{vram} GB VRAM is tight for a 7-8B QLoRA; keep lora_max_seq low "
                     "(<=1024), batch 1, and expect a slow run — or set a 3B base")
    if "bitsandbytes" not in missing and vram is not None:
        notes.append("bitsandbytes on Windows can be finicky; if 4-bit fails, use WSL2")
    ok = (not missing) and cuda and (vram is not None)
    return Feasibility(ok=ok, gpu=gpu, vram_gb=vram, cuda=cuda,
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


TEACHER_TASKS = [
    "Explain the difference between correlation and causation with one concrete example.",
    "Write a Python function that returns the nth Fibonacci number iteratively.",
    "Summarize the tradeoffs between TCP and UDP in three bullet points.",
    "Given costs [4,2,7] and values [3,1,5], which item has the best value-per-cost?",
    "Rewrite this to be concise: 'It is important to note that in order to...'.",
    "What are the main risks of running arbitrary code from the internet?",
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
    _emit(emit, "lora", f"loading base {cfg.lora_base_model} in "
          f"{'4-bit' if cfg.lora_4bit else 'full'} precision", level="tool")
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (AutoModelForCausalLM, AutoTokenizer,
                                  BitsAndBytesConfig, DataCollatorForLanguageModeling,
                                  Trainer, TrainingArguments)

        tok = AutoTokenizer.from_pretrained(cfg.lora_base_model, use_fast=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        quant = None
        if cfg.lora_4bit:
            quant = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(
            cfg.lora_base_model, quantization_config=quant, device_map="auto",
            torch_dtype=torch.float16)
        model = prepare_model_for_kbit_training(model)
        lconf = LoraConfig(
            r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
            bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
        model = get_peft_model(model, lconf)

        ds = load_dataset("json", data_files=str(DATASET_FILE), split="train")
        ds = ds.map(lambda r: _format_example(tok, r["instruction"], r["output"],
                                              cfg.lora_max_seq),
                    remove_columns=ds.column_names)
        out_dir = ADAPTERS_DIR / time.strftime("%Y%m%d-%H%M%S")
        args = TrainingArguments(
            output_dir=str(out_dir / "_trainer"),
            per_device_train_batch_size=cfg.lora_batch_size,
            gradient_accumulation_steps=cfg.lora_grad_accum,
            num_train_epochs=cfg.lora_epochs, learning_rate=cfg.lora_lr,
            fp16=True, logging_steps=5, save_strategy="no", report_to=[])
        trainer = Trainer(
            model=model, args=args, train_dataset=ds,
            data_collator=DataCollatorForLanguageModeling(tok, mlm=False))
        _emit(emit, "lora", f"training on {n} examples "
              f"({cfg.lora_epochs} epoch(s))", level="tool")
        trainer.train()
        out_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(out_dir))
        tok.save_pretrained(str(out_dir))
        (out_dir / "ag_meta.json").write_text(json.dumps({
            "base": cfg.lora_base_model, "examples": n, "r": cfg.lora_r,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=2), encoding="utf-8")
        _emit(emit, "lora", f"adapter saved to {out_dir}", level="result")
        return TrainResult(True, adapter_path=str(out_dir), examples=n,
                           reason="training complete")
    except Exception as e:  # pragma: no cover - depends on GPU/deps at runtime
        return TrainResult(False, reason=f"training failed: {e}")
