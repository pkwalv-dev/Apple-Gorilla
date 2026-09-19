# LoRA training on Windows via WSL2 (verified)

The verified setup for QLoRA-fine-tuning AG's local model on an NVIDIA GPU under Windows.
Training runs inside **WSL2** (Linux CUDA userspace); the rest of AG can stay on Windows.

## Why WSL2

- Windows' system Python may be too new for the pinned CUDA stack; a WSL2 **uv** venv on
  **Python 3.12** gives a clean, reproducible environment.
- `bitsandbytes` and Unsloth's Triton kernels target Linux CUDA and are far less finicky
  there than on native Windows.

## The stack (verified combo)

Managed by `uv` in `~/ag-lora/.venv` (Python 3.12):

- `torch 2.6.0+cu124` (CUDA + bf16 on the GPU), `transformers 5.5.0`, `peft 0.21.0`,
  `bitsandbytes 0.50.2`, `unsloth 2026.9.6`, `datasets`.
- **Pin `torchao==0.13.0`.** torchao ≥0.17 calls `torch.utils._pytree.register_constant`,
  which only exists in torch ≥2.7; on torch 2.6 a newer torchao breaks the import path.
  (See the merge caveat below for the one place 0.13 is temporarily removed.)

## The C compiler (required for Unsloth/Triton)

Triton JIT-compiles kernels at the first training step and needs a host **C compiler**.
A minimal WSL Ubuntu has none, and `apt install build-essential` needs root. The no-root
fix used here: a **conda-forge gcc toolchain in userspace** via micromamba —

- installed at `~/ag-lora/buildenv` (gcc/g++/sysroot), with stable `gcc`/`cc`/`g++`
  symlinks;
- wired into training through `CC`/`CXX` (see the launcher).

Without it, `lora_use_unsloth=true` crashes at step 0 (“Failed to find C compiler”); the
transformers+peft fallback works without a compiler but is ~15× slower per step.

## The launcher

`~/ag-lora/ag-train.sh` runs AG's `lora` commands inside the venv against the Windows repo:

```bash
export HF_HOME="$HOME/ag-lora/hf-cache"
export VIRTUAL_ENV="$HOME/ag-lora/.venv"
export PATH="$HOME/ag-lora/buildenv/bin:$PATH"      # userspace gcc
export CC="$HOME/ag-lora/buildenv/bin/gcc"
export CXX="$HOME/ag-lora/buildenv/bin/g++"
cd /mnt/c/Users/<you>/…/Apple-Gorilla
exec "$HOME/ag-lora/.venv/bin/python" -m ag "$@"
```

## Windows ↔ WSL split

The repo lives on the Windows drive (`/mnt/c/...`), so both sides share `config.json` and
`state/lora/`:

- **`build-data`** needs Claude credentials (the teacher set) but no GPU → run on
  **Windows**: `python -m ag lora build-data`.
- **`train`** needs the GPU → run in **WSL**: `~/ag-lora/ag-train.sh lora train`.
- Give a 7B fp16 merge room: WSL defaults to ~half of host RAM; raise it in
  `%USERPROFILE%\.wslconfig` (`[wsl2]` `memory=24GB`) and `wsl --shutdown` to apply.

## Merge → GGUF → Ollama (closing the loop)

`ag lora merge <id>` merges the adapter into its base, converts to GGUF (needs
llama.cpp's `convert_hf_to_gguf.py`), and registers it with Ollama as a selectable model.

- Build llama.cpp's `llama-quantize` with the same userspace gcc (cmake + ninja via
  `uv pip`); convert to f16 then quantize to `Q4_K_M` for an 8GB card.
- **torchao caveat:** this `peft` raises on torchao 0.13, but the fp16 merge doesn't need
  torchao at all and peft skips it when absent. Temporarily `uv pip uninstall torchao`
  for the merge, then reinstall `torchao==0.13.0` so training keeps working.
- Because Ollama runs on the Windows host, do the `ollama create` step on Windows (it can
  read the GGUF from the shared `state/lora/...` path).

## Quick reference

```bash
~/ag-lora/ag-train.sh lora status     # GPU/VRAM/CUDA + Unsloth + deps + can-train?
python -m ag lora build-data          # Windows: teacher + memory → dataset.jsonl
~/ag-lora/ag-train.sh lora train      # WSL: the QLoRA run
```
